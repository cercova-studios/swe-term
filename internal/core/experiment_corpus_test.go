package core

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The experiment manifests freeze only their fixtures directory, so each
// fixtures directory carries an executable-corpus.sha256 lock (sha256sum
// format, repository-relative paths) that pins the Go test files which are the
// executable form of the trace corpus. The lock is inside the digested
// directory, which makes the manifest content_digest cover the executable
// corpus transitively. This test fails closed when a pinned file changes
// without a new corpus revision, lock, and manifest digest.
func TestFrozenExperimentCorpusMatchesLock(t *testing.T) {
	repoRoot := filepath.Join("..", "..")
	for _, id := range []string{"evidence-gated-lifecycle", "temporal-journal-monitor"} {
		t.Run(id, func(t *testing.T) {
			lockPath := filepath.Join(repoRoot, "experiments", "specs", id, "fixtures", "executable-corpus.sha256")
			lock, err := os.ReadFile(lockPath)
			if err != nil {
				t.Fatal(err)
			}
			pinned := 0
			for _, line := range strings.Split(string(lock), "\n") {
				fields := strings.Fields(line)
				if len(fields) == 0 {
					continue
				}
				if len(fields) != 2 {
					t.Fatalf("%s: malformed lock line %q", lockPath, line)
				}
				want, path := fields[0], fields[1]
				data, err := os.ReadFile(filepath.Join(repoRoot, filepath.FromSlash(path)))
				if err != nil {
					t.Fatal(err)
				}
				sum := sha256.Sum256(data)
				if got := hex.EncodeToString(sum[:]); got != want {
					t.Errorf("%s changed without a new corpus revision: lock has %s, file is %s", path, want, got)
				}
				pinned++
			}
			if pinned == 0 {
				t.Fatalf("%s pins no executable corpus files", lockPath)
			}
		})
	}
}
