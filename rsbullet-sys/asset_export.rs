//! Copying Bullet's example data into the user's data directory: the build
//! script's export, kept apart so that `tests/asset_export.rs` runs it.
//!
//! The export used to copy each source tree in turn and keep any target
//! directory that existed. Both went wrong on a read-only source tree
//! (a package cache, a store, a checkout made read-only): `fs::copy` gives
//! the copy the source's permissions, so the second tree's file of a name
//! the first already wrote could not be opened, the export stopped halfway,
//! and the half-filled directory was then kept for good.

use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
};

/// What an export did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Exported {
    /// Files written: missing, or of another size than their source.
    pub copied: usize,
    /// Files already there with their source's size.
    pub kept: usize,
}

/// Makes `target` hold every file of the `sources` trees under its path
/// relative to its tree; where several trees have a path, the last one's
/// file. A file already in `target` with the size of its source is kept,
/// made writable by its owner if an earlier export left it read-only; any
/// other is written, whatever its permissions, and left writable by its
/// owner. Files in `target` that no source has are left alone.
///
/// Each file is copied to a temporary name beside it and renamed into
/// place, so an interrupted export leaves whole files, old or new, and the
/// next one finishes it; two builds exporting at once write the same files.
pub fn export(sources: &[&Path], target: &Path) -> Result<Exported, String> {
    let mut files = BTreeMap::new();
    for source in sources {
        list(source, Path::new(""), &mut files)?;
    }
    let mut done = Exported::default();
    for (relative, from) in &files {
        let to = target.join(relative);
        let size = fs::metadata(from)
            .map_err(|e| failed("read", from, e))?
            .len();
        let present = fs::symlink_metadata(&to)
            .ok()
            .filter(|m| m.is_file() && m.len() == size);
        if let Some(present) = present {
            if present.permissions().readonly() {
                writable(&to)?;
            }
            done.kept += 1;
            continue;
        }
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent).map_err(|e| failed("create", parent, e))?;
        }
        copy(from, &to)?;
        done.copied += 1;
    }
    Ok(done)
}

/// Every file under `root.join(relative)`, by its path relative to `root`.
fn list(
    root: &Path,
    relative: &Path,
    files: &mut BTreeMap<PathBuf, PathBuf>,
) -> Result<(), String> {
    let directory = root.join(relative);
    let entries = fs::read_dir(&directory).map_err(|e| failed("read", &directory, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| failed("read", &directory, e))?;
        let (path, relative) = (entry.path(), relative.join(entry.file_name()));
        if path.is_dir() {
            list(root, &relative, files)?;
        } else {
            files.insert(relative, path);
        }
    }
    Ok(())
}

/// Copies `from` over `to` by way of a temporary file beside `to`.
fn copy(from: &Path, to: &Path) -> Result<(), String> {
    let name = to.file_name().unwrap_or_default().to_string_lossy();
    let partial = to.with_file_name(format!(".{name}.part-{}", std::process::id()));
    let result = fs::copy(from, &partial)
        .map_err(|e| failed("copy", from, e))
        .and_then(|_| writable(&partial))
        // Windows refuses to rename over a read-only file.
        .and_then(|()| match fs::symlink_metadata(to) {
            Ok(m) if m.is_file() => writable(to),
            _ => Ok(()),
        })
        .and_then(|()| fs::rename(&partial, to).map_err(|e| failed("replace", to, e)));
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

/// Lets the owner write `path`: `fs::copy` gave it the source's
/// permissions, read-only for a read-only tree.
fn writable(path: &Path) -> Result<(), String> {
    let mut permissions = fs::metadata(path)
        .map_err(|e| failed("read", path, e))?
        .permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode(permissions.mode() | 0o200);
    }
    #[cfg(not(unix))]
    #[allow(
        clippy::permissions_set_readonly_false,
        reason = "outside Unix the read-only attribute is all there is to clear"
    )]
    permissions.set_readonly(false);
    fs::set_permissions(path, permissions).map_err(|e| failed("make writable", path, e))
}

fn failed(what: &str, path: &Path, error: io::Error) -> String {
    format!("failed to {what} {}: {error}", path.display())
}
