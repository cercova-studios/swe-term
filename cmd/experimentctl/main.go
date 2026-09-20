package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const manifestSchemaVersion = 1

var (
	experimentIDPattern    = regexp.MustCompile(`^[a-z][a-z0-9-]{2,63}$`)
	variantNamePattern     = regexp.MustCompile(`^[a-z][a-z0-9-]{1,31}$`)
	environmentNamePattern = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
	digestPattern          = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

type manifest struct {
	SchemaVersion       int         `json:"schema_version"`
	ExperimentID        string      `json:"experiment_id"`
	Status              string      `json:"status"`
	Kind                string      `json:"kind"`
	Papers              []paper     `json:"papers"`
	DesignReferences    []paper     `json:"design_references"`
	Hypothesis          string      `json:"hypothesis"`
	NullHypothesis      string      `json:"null_hypothesis"`
	IndependentVariable string      `json:"independent_variable"`
	Source              source      `json:"source"`
	Environment         environment `json:"environment"`
	Model               model       `json:"model"`
	Variants            []variant   `json:"variants"`
	Repetitions         int         `json:"repetitions"`
	Randomness          randomness  `json:"randomness"`
	Metrics             metrics     `json:"metrics"`
	Evaluator           evaluator   `json:"evaluator"`
	Evidence            evidence    `json:"evidence"`
}

type paper struct {
	Title string `json:"title"`
	URL   string `json:"url"`
	Claim string `json:"claim"`
}

type source struct {
	Kind          string `json:"kind"`
	Locator       string `json:"locator"`
	Revision      string `json:"revision"`
	ContentDigest string `json:"content_digest"`
}

type environment struct {
	Adapter              string    `json:"adapter"`
	Network              string    `json:"network"`
	NetworkJustification string    `json:"network_justification"`
	AllowedEnv           []string  `json:"allowed_env"`
	Resources            resources `json:"resources"`
}

type resources struct {
	TimeoutSeconds int    `json:"timeout_seconds"`
	MaxParallelism int    `json:"max_parallelism"`
	MaxOutputBytes int64  `json:"max_output_bytes"`
	MemoryBytes    *int64 `json:"memory_bytes"`
	CPUSeconds     *int   `json:"cpu_seconds"`
}

type model struct {
	Required         bool           `json:"required"`
	Provider         string         `json:"provider"`
	ID               string         `json:"id"`
	Parameters       map[string]any `json:"parameters"`
	PromptDigest     string         `json:"prompt_digest"`
	ToolSchemaDigest string         `json:"tool_schema_digest"`
}

type variant struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Command     []string       `json:"command"`
	Config      map[string]any `json:"config"`
}

type randomness struct {
	Mode          string `json:"mode"`
	Seeds         []int  `json:"seeds"`
	Justification string `json:"justification"`
}

type metrics struct {
	Primary        string   `json:"primary"`
	Secondary      []string `json:"secondary"`
	AcceptanceRule string   `json:"acceptance_rule"`
}

type evaluator struct {
	Kind         string   `json:"kind"`
	Command      []string `json:"command"`
	Rubric       string   `json:"rubric"`
	RubricDigest string   `json:"rubric_digest"`
}

type evidence struct {
	RawRuns string `json:"raw_runs"`
	Curated string `json:"curated"`
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		printUsage(stderr)
		return 2
	}

	root, err := findRepositoryRoot()
	if err != nil {
		fmt.Fprintf(stderr, "experimentctl: %v\n", err)
		return 1
	}

	switch args[0] {
	case "new":
		if len(args) != 2 {
			fmt.Fprintln(stderr, "usage: experimentctl new <experiment-id>")
			return 2
		}
		created, err := newExperiment(root, args[1])
		if err != nil {
			fmt.Fprintf(stderr, "experimentctl new: %v\n", err)
			return 1
		}
		if created {
			fmt.Fprintf(stdout, "created experiments/specs/%s\n", args[1])
		} else {
			fmt.Fprintf(stdout, "already exists: experiments/specs/%s\n", args[1])
		}
		return 0
	case "validate", "ready":
		if len(args) != 2 {
			fmt.Fprintf(stderr, "usage: experimentctl %s <experiment-id>\n", args[0])
			return 2
		}
		ready := args[0] == "ready"
		errs := validateExperiment(root, args[1], ready)
		if len(errs) > 0 {
			for _, validationErr := range errs {
				fmt.Fprintf(stderr, "- %s\n", validationErr)
			}
			return 1
		}
		gate := "draft"
		if ready {
			gate = "preregistration"
		}
		fmt.Fprintf(stdout, "%s passes the %s gate\n", args[1], gate)
		return 0
	case "digest":
		if len(args) != 2 {
			fmt.Fprintln(stderr, "usage: experimentctl digest <experiment-id>")
			return 2
		}
		digest, err := digestExperimentSource(root, args[1])
		if err != nil {
			fmt.Fprintf(stderr, "experimentctl digest: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, digest)
		return 0
	case "list":
		if len(args) != 1 {
			fmt.Fprintln(stderr, "usage: experimentctl list")
			return 2
		}
		if err := listExperiments(root, stdout); err != nil {
			fmt.Fprintf(stderr, "experimentctl list: %v\n", err)
			return 1
		}
		return 0
	default:
		fmt.Fprintf(stderr, "unknown command %q\n", args[0])
		printUsage(stderr)
		return 2
	}
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: experimentctl <new|validate|ready|digest|list> [experiment-id]")
}

func findRepositoryRoot() (string, error) {
	current, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if fileExists(filepath.Join(current, "go.mod")) && fileExists(filepath.Join(current, "experiments", "templates", "manifest.json")) {
			return current, nil
		}
		parent := filepath.Dir(current)
		if parent == current {
			return "", errors.New("not inside the swe-term repository")
		}
		current = parent
	}
}

func newExperiment(root, id string) (created bool, err error) {
	if !experimentIDPattern.MatchString(id) {
		return false, fmt.Errorf("invalid experiment id %q; use 3-64 lowercase letters, digits, and hyphens, starting with a letter", id)
	}

	target := filepath.Join(root, "experiments", "specs", id)
	manifestPath := filepath.Join(target, "manifest.json")
	readmePath := filepath.Join(target, "README.md")
	fixturesPath := filepath.Join(target, "fixtures", "README.md")
	if _, statErr := os.Stat(target); statErr == nil {
		if fileExists(manifestPath) && fileExists(readmePath) && fileExists(fixturesPath) {
			return false, nil
		}
		return false, fmt.Errorf("refusing to modify partial experiment directory %s", target)
	} else if !errors.Is(statErr, os.ErrNotExist) {
		return false, statErr
	}

	if err := os.MkdirAll(filepath.Join(target, "fixtures"), 0o755); err != nil {
		return false, err
	}
	complete := false
	defer func() {
		if !complete {
			_ = os.RemoveAll(target)
		}
	}()

	templates := []struct {
		from string
		to   string
	}{
		{from: filepath.Join(root, "experiments", "templates", "manifest.json"), to: manifestPath},
		{from: filepath.Join(root, "experiments", "templates", "README.md"), to: readmePath},
		{from: filepath.Join(root, "experiments", "templates", "FIXTURES.md"), to: fixturesPath},
	}
	for _, item := range templates {
		if err := instantiateTemplate(item.from, item.to, id); err != nil {
			return false, err
		}
	}

	complete = true
	return true, nil
}

func instantiateTemplate(sourcePath, targetPath, id string) error {
	content, err := os.ReadFile(sourcePath)
	if err != nil {
		return err
	}
	content = bytes.ReplaceAll(content, []byte("__EXPERIMENT_ID__"), []byte(id))
	file, err := os.OpenFile(targetPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	if _, err := file.Write(content); err != nil {
		_ = file.Close()
		return err
	}
	return file.Close()
}

func validateExperiment(root, id string, ready bool) []string {
	if !experimentIDPattern.MatchString(id) {
		return []string{fmt.Sprintf("invalid experiment id %q", id)}
	}
	path := filepath.Join(root, "experiments", "specs", id, "manifest.json")
	m, err := decodeManifest(path)
	if err != nil {
		return []string{err.Error()}
	}
	errs := validateManifest(m, id, ready)
	if ready && m.Source.Kind == "directory" && digestPattern.MatchString(m.Source.ContentDigest) {
		digest, digestErr := digestExperimentSource(root, id)
		if digestErr != nil {
			errs = append(errs, "source digest: "+digestErr.Error())
		} else if digest != m.Source.ContentDigest {
			errs = append(errs, fmt.Sprintf("source.content_digest mismatch: manifest has %s, fixtures are %s", m.Source.ContentDigest, digest))
		}
	}
	sort.Strings(errs)
	return errs
}

func digestExperimentSource(root, id string) (string, error) {
	if !experimentIDPattern.MatchString(id) {
		return "", fmt.Errorf("invalid experiment id %q", id)
	}
	manifestPath := filepath.Join(root, "experiments", "specs", id, "manifest.json")
	m, err := decodeManifest(manifestPath)
	if err != nil {
		return "", err
	}
	if m.Source.Kind != "directory" {
		return "", fmt.Errorf("source.kind %q does not use the directory digest adapter", m.Source.Kind)
	}
	if !isSafeRelativePath(m.Source.Locator) {
		return "", errors.New("source.locator must be a safe path relative to the experiment specification")
	}
	specificationRoot := filepath.Join(root, "experiments", "specs", id)
	sourcePath := filepath.Join(specificationRoot, filepath.Clean(m.Source.Locator))
	if !pathWithin(specificationRoot, sourcePath) {
		return "", errors.New("source.locator escapes the experiment specification")
	}
	return digestDirectory(sourcePath)
}

func decodeManifest(path string) (manifest, error) {
	file, err := os.Open(path)
	if err != nil {
		return manifest{}, err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var m manifest
	if err := decoder.Decode(&m); err != nil {
		return manifest{}, fmt.Errorf("decode %s: %w", path, err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return manifest{}, fmt.Errorf("decode %s: multiple JSON values", path)
		}
		return manifest{}, fmt.Errorf("decode %s: %w", path, err)
	}
	return m, nil
}

func validateManifest(m manifest, expectedID string, ready bool) []string {
	var errs []string
	add := func(condition bool, message string) {
		if !condition {
			errs = append(errs, message)
		}
	}

	add(m.SchemaVersion == manifestSchemaVersion, fmt.Sprintf("schema_version must be %d", manifestSchemaVersion))
	add(experimentIDPattern.MatchString(m.ExperimentID), "experiment_id has an invalid format")
	add(m.ExperimentID == expectedID, "experiment_id must match its specification directory")
	add(contains([]string{"draft", "preregistered", "running", "complete", "rejected"}, m.Status), "status is unsupported")
	add(contains([]string{"mechanism-hypothesis", "benchmark"}, m.Kind), "kind is unsupported")
	add(contains([]string{"directory", "generated", "archive", "jj", "git"}, m.Source.Kind), "source.kind is unsupported")
	add(contains([]string{"process", "container", "external"}, m.Environment.Adapter), "environment.adapter is unsupported")
	add(contains([]string{"disabled", "loopback", "inherit"}, m.Environment.Network), "environment.network is unsupported")
	add(m.Environment.Resources.TimeoutSeconds > 0, "environment.resources.timeout_seconds must be positive")
	add(m.Environment.Resources.MaxParallelism > 0, "environment.resources.max_parallelism must be positive")
	add(m.Environment.Resources.MaxOutputBytes >= 1024, "environment.resources.max_output_bytes must be at least 1024")
	if m.Environment.Resources.MemoryBytes != nil {
		add(*m.Environment.Resources.MemoryBytes > 0, "environment.resources.memory_bytes must be positive when set")
	}
	if m.Environment.Resources.CPUSeconds != nil {
		add(*m.Environment.Resources.CPUSeconds > 0, "environment.resources.cpu_seconds must be positive when set")
	}

	seenEnv := make(map[string]struct{}, len(m.Environment.AllowedEnv))
	for _, name := range m.Environment.AllowedEnv {
		add(environmentNamePattern.MatchString(name), fmt.Sprintf("allowed environment name %q is invalid", name))
		if _, exists := seenEnv[name]; exists {
			errs = append(errs, fmt.Sprintf("allowed environment name %q is duplicated", name))
		}
		seenEnv[name] = struct{}{}
	}

	add(len(m.Variants) >= 2, "at least two variants are required")
	seenVariants := make(map[string]struct{}, len(m.Variants))
	for _, item := range m.Variants {
		add(variantNamePattern.MatchString(item.Name), fmt.Sprintf("variant name %q is invalid", item.Name))
		if _, exists := seenVariants[item.Name]; exists {
			errs = append(errs, fmt.Sprintf("variant name %q is duplicated", item.Name))
		}
		seenVariants[item.Name] = struct{}{}
	}
	add(m.Repetitions >= 1, "repetitions must be positive")
	add(contains([]string{"deterministic", "seeded", "uncontrolled"}, m.Randomness.Mode), "randomness.mode is unsupported")
	seenSeeds := make(map[int]struct{}, len(m.Randomness.Seeds))
	for _, seed := range m.Randomness.Seeds {
		if _, exists := seenSeeds[seed]; exists {
			errs = append(errs, fmt.Sprintf("randomness seed %d is duplicated", seed))
		}
		seenSeeds[seed] = struct{}{}
	}
	add(contains([]string{"manual", "command", "mixed"}, m.Evaluator.Kind), "evaluator.kind is unsupported")
	validateEvidencePath := func(label, value, expected string) {
		add(isSafeRelativePath(value), label+" must be a safe repository-relative path")
		add(filepath.ToSlash(filepath.Clean(value)) == expected, label+" must be "+expected)
	}
	validateEvidencePath("evidence.raw_runs", m.Evidence.RawRuns, "experiments/runs/"+expectedID)
	validateEvidencePath("evidence.curated", m.Evidence.Curated, "experiments/evidence/"+expectedID)

	if !ready {
		sort.Strings(errs)
		return errs
	}

	add(m.Status == "preregistered", "status must be preregistered before a result-producing run")
	if m.Kind == "mechanism-hypothesis" {
		add(len(m.Papers) > 0, "at least one paper claim is required for mechanism-hypothesis experiments")
	} else {
		add(len(m.DesignReferences) > 0, "at least one design reference is required for benchmark experiments")
	}
	for i, item := range m.Papers {
		prefix := fmt.Sprintf("papers[%d]", i)
		add(strings.TrimSpace(item.Title) != "", prefix+".title is required")
		add(validHTTPURL(item.URL), prefix+".url must be an absolute http or https URL")
		add(strings.TrimSpace(item.Claim) != "", prefix+".claim is required")
	}
	for i, item := range m.DesignReferences {
		prefix := fmt.Sprintf("design_references[%d]", i)
		add(strings.TrimSpace(item.Title) != "", prefix+".title is required")
		add(validHTTPURL(item.URL), prefix+".url must be an absolute http or https URL")
		add(strings.TrimSpace(item.Claim) != "", prefix+".claim is required")
	}
	add(strings.TrimSpace(m.Hypothesis) != "", "hypothesis is required")
	add(strings.TrimSpace(m.NullHypothesis) != "", "null_hypothesis is required")
	add(strings.TrimSpace(m.IndependentVariable) != "", "independent_variable is required")
	add(strings.TrimSpace(m.Source.Locator) != "", "source.locator is required")
	add(digestPattern.MatchString(m.Source.ContentDigest), "source.content_digest must be sha256:<64 lowercase hex characters>")
	if contains([]string{"directory", "generated"}, m.Source.Kind) {
		add(isSafeRelativePath(m.Source.Locator), "source.locator must be a safe relative path for directory and generated sources")
	}
	if contains([]string{"archive", "jj", "git"}, m.Source.Kind) {
		add(strings.TrimSpace(m.Source.Revision) != "", "source.revision is required for archive, jj, and git sources")
	}
	if m.Environment.Network != "disabled" {
		add(strings.TrimSpace(m.Environment.NetworkJustification) != "", "network_justification is required when network is not disabled")
	}

	for i, item := range m.Variants {
		prefix := fmt.Sprintf("variants[%d]", i)
		add(strings.TrimSpace(item.Description) != "", prefix+".description is required")
		add(len(item.Command) > 0, prefix+".command must be a non-empty argv array")
		for j, arg := range item.Command {
			add(arg != "", fmt.Sprintf("%s.command[%d] must not be empty", prefix, j))
		}
	}
	if m.Model.Required {
		add(strings.TrimSpace(m.Model.Provider) != "", "model.provider is required for model-dependent experiments")
		add(strings.TrimSpace(m.Model.ID) != "", "model.id is required for model-dependent experiments")
		add(digestPattern.MatchString(m.Model.PromptDigest), "model.prompt_digest must be a sha256 digest")
		add(digestPattern.MatchString(m.Model.ToolSchemaDigest), "model.tool_schema_digest must be a sha256 digest")
		add(m.Repetitions >= 3, "model-dependent experiments require at least three repetitions")
	}
	switch m.Randomness.Mode {
	case "deterministic":
		add(len(m.Randomness.Seeds) == 0, "deterministic randomness mode must not declare seeds")
	case "seeded":
		add(len(m.Randomness.Seeds) == m.Repetitions, "seeded randomness mode requires one unique seed per repetition")
	case "uncontrolled":
		add(len(m.Randomness.Seeds) == 0, "uncontrolled randomness mode must not declare seeds")
		add(strings.TrimSpace(m.Randomness.Justification) != "", "uncontrolled randomness requires a justification")
	}
	add(strings.TrimSpace(m.Metrics.Primary) != "", "metrics.primary is required")
	add(strings.TrimSpace(m.Metrics.AcceptanceRule) != "", "metrics.acceptance_rule is required")
	add(strings.TrimSpace(m.Evaluator.Rubric) != "", "evaluator.rubric is required")
	add(digestPattern.MatchString(m.Evaluator.RubricDigest), "evaluator.rubric_digest must be a sha256 digest")
	if contains([]string{"command", "mixed"}, m.Evaluator.Kind) {
		add(len(m.Evaluator.Command) > 0, "evaluator.command must be set for command and mixed evaluators")
	}

	sort.Strings(errs)
	return errs
}

func listExperiments(root string, stdout io.Writer) error {
	directory := filepath.Join(root, "experiments", "specs")
	entries, err := os.ReadDir(directory)
	if err != nil {
		return err
	}
	found := false
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		found = true
		path := filepath.Join(directory, entry.Name(), "manifest.json")
		m, decodeErr := decodeManifest(path)
		if decodeErr != nil {
			fmt.Fprintf(stdout, "%s\tinvalid\t%s\n", entry.Name(), decodeErr)
			continue
		}
		fmt.Fprintf(stdout, "%s\t%s\n", entry.Name(), m.Status)
	}
	if !found {
		fmt.Fprintln(stdout, "no experiment specifications")
	}
	return nil
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func contains(values []string, candidate string) bool {
	for _, value := range values {
		if value == candidate {
			return true
		}
	}
	return false
}

func isSafeRelativePath(value string) bool {
	if value == "" || filepath.IsAbs(value) {
		return false
	}
	cleaned := filepath.Clean(value)
	return cleaned != "." && cleaned != ".." && !strings.HasPrefix(cleaned, ".."+string(filepath.Separator))
}

func pathWithin(parent, candidate string) bool {
	relative, err := filepath.Rel(parent, candidate)
	return err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator))
}

func digestDirectory(root string) (string, error) {
	info, err := os.Stat(root)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%s is not a directory", root)
	}

	var files []string
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("fixture symlinks are not supported: %s", path)
		}
		if entry.IsDir() {
			return nil
		}
		if !entry.Type().IsRegular() {
			return fmt.Errorf("fixture is not a regular file: %s", path)
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		return "", err
	}
	sort.Strings(files)

	hash := sha256.New()
	for _, path := range files {
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return "", err
		}
		relative = filepath.ToSlash(relative)
		info, err := os.Stat(path)
		if err != nil {
			return "", err
		}
		if err := binary.Write(hash, binary.BigEndian, uint64(len(relative))); err != nil {
			return "", err
		}
		if _, err := io.WriteString(hash, relative); err != nil {
			return "", err
		}
		executable := byte(0)
		if info.Mode().Perm()&0o111 != 0 {
			executable = 1
		}
		if _, err := hash.Write([]byte{executable}); err != nil {
			return "", err
		}
		if err := binary.Write(hash, binary.BigEndian, uint64(info.Size())); err != nil {
			return "", err
		}
		file, err := os.Open(path)
		if err != nil {
			return "", err
		}
		_, copyErr := io.Copy(hash, file)
		closeErr := file.Close()
		if copyErr != nil {
			return "", copyErr
		}
		if closeErr != nil {
			return "", closeErr
		}
	}
	return "sha256:" + hex.EncodeToString(hash.Sum(nil)), nil
}

func validHTTPURL(value string) bool {
	parsed, err := url.Parse(value)
	return err == nil && (parsed.Scheme == "http" || parsed.Scheme == "https") && parsed.Host != "" && parsed.User == nil
}
