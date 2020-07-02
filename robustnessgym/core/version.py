from types import SimpleNamespace

import dill as pickle
from semver import VersionInfo as Version


class SemanticVersionerMixin:
    """Simple mixin that adds semantic versioning to any class."""

    def __init__(self, version: str = "0.0.1", *args, **kwargs):
        super(SemanticVersionerMixin, self).__init__(*args, **kwargs)
        self._version = Version.parse(version)
        self._version_history = {}
        self._last_digest = None

        # TODO(karan): implement more features for commit-then-bump, add diffing

    @property
    def version(self):
        return str(self._version)

    @property
    def version_history(self):
        return self._version_history

    @property
    def major(self):
        return self._version.major

    @property
    def minor(self):
        return self._version.minor

    @property
    def patch(self):
        return self._version.patch

    def bump_major(self):
        """Commit the current version and bump the major version."""
        self.commit()
        self._version = self._version.bump_major()
        self._last_digest = self.digest()

    def bump_minor(self):
        """Commit the current version and bump the minor version."""
        self.commit()
        self._version = self._version.bump_minor()
        self._last_digest = self.digest()

    def bump_patch(self):
        """Commit the current version and bump the patch."""
        self.commit()
        self._version = self._version.bump_major()
        self._last_digest = self.digest()

    def commit(self):
        """Commit the current version to history.

        Multiple commits on the same version overwrite each other.
        """
        self._version_history[self.version] = self.digest()

    def digest(self) -> str:
        """Compute a digest for the object."""
        raise NotImplementedError(
            "Must implement a digest for the object that is being versioned."
        )

    def diff(self, digest: str, otherdigest: str) -> bool:
        """Check if digests have changed."""
        return digest != otherdigest

    def _dumps_version(self) -> str:
        return pickle.dumps(
            SimpleNamespace(
                version=self.version,
                history=self._version_history,
                last_digest=self._last_digest,
            )
        )

    def _loads_version(self, s: str):
        namespace = pickle.loads(s)
        self._version = namespace.version
        self._version_history = namespace.history
        self._last_digest = namespace.last_digest
