"""Model version management for tracking multiple training runs."""

import json
import os
from pathlib import Path
from datetime import datetime


class VersionManager:
    """Manages model versions and output directories for multiple training runs."""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        self.base_dir = Path(base_dir)
        self.version_file = self.base_dir / 'models' / 'version_registry.json'
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        
    def get_next_version(self):
        """Get the next available version number."""
        registry = self._load_registry()
        if not registry:
            return 1
        return max(int(v) for v in registry.keys()) + 1
    
    def register_version(self, version_num, metadata):
        """Register a new model version with metadata."""
        registry = self._load_registry()
        registry[str(version_num)] = {
            'timestamp': datetime.now().isoformat(),
            **metadata
        }
        self._save_registry(registry)
    
    def get_version_info(self, version_num):
        """Get metadata for a specific version."""
        registry = self._load_registry()
        return registry.get(str(version_num))
    
    def create_version_dirs(self, version_num):
        """Create directory structure for a model version."""
        dirs = {
            'model': self.base_dir / 'models' / 'best' / f'v{version_num}',
            'checkpoints': self.base_dir / 'models' / 'checkpoints' / f'v{version_num}',
            'outputs': self.base_dir / 'outputs' / f'v{version_num}',
            'mispredictions': self.base_dir / 'outputs' / f'v{version_num}' / 'mispredictions',
            'influence': self.base_dir / 'outputs' / f'v{version_num}' / 'influence_analysis',
            'inspection': self.base_dir / 'outputs' / f'v{version_num}' / 'inspection',
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return dirs
    
    def _load_registry(self):
        """Load version registry from JSON file."""
        if not self.version_file.exists():
            return {}
        with open(self.version_file, 'r') as f:
            return json.load(f)
    
    def _save_registry(self, registry):
        """Save version registry to JSON file."""
        with open(self.version_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def list_versions(self):
        """List all registered versions."""
        registry = self._load_registry()
        return sorted([int(v) for v in registry.keys()])
    
    def get_latest_version(self):
        """Get the most recent version number."""
        versions = self.list_versions()
        return versions[-1] if versions else None
