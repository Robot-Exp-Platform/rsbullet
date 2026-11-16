//! The build script's asset export (`asset_export.rs`) from read-only
//! source trees, as a package cache or a read-only checkout gives it: two
//! trees with a file of the same name export in full, the later tree's file
//! winning, and every file written is writable; a target directory an
//! earlier export left half done is completed.
//!
//! Before, the copy kept the source's read-only mode, so the second tree's
//! file of a name the first had written could not be opened and the export
//! stopped there; and an existing target was kept whole, however little it
//! held.

use std::{
    fs,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
};

#[path = "../asset_export.rs"]
mod asset_export;

use asset_export::{Exported, export};

/// Two source trees, as `bullet3/examples/pybullet/gym/pybullet_data` and
/// `bullet3/data` are: `shared.urdf` and `meshes/shared.obj` are in both.
const FIRST: [(&str, &str); 4] = [
    ("a.txt", "first a"),
    ("shared.urdf", "first"),
    ("meshes/x.obj", "x from the first tree"),
    ("meshes/shared.obj", "first mesh"),
];
const SECOND: [(&str, &str); 4] = [
    ("b.txt", "second b"),
    ("shared.urdf", "second, longer"),
    ("meshes/y.obj", "y"),
    ("meshes/shared.obj", "second mesh, longer"),
];

/// The export of `FIRST` then `SECOND`.
fn expected() -> Vec<(&'static str, &'static str)> {
    vec![
        ("a.txt", "first a"),
        ("b.txt", "second b"),
        ("meshes/shared.obj", "second mesh, longer"),
        ("meshes/x.obj", "x from the first tree"),
        ("meshes/y.obj", "y"),
        ("shared.urdf", "second, longer"),
    ]
}

/// A directory of the test's own, removed first if an earlier run left it.
fn scratch(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "rsbullet-sys-asset-export-{name}-{}",
        std::process::id()
    ));
    if path.exists() {
        set_read_only(&path, false);
        fs::remove_dir_all(&path).unwrap();
    }
    path
}

fn write_tree(root: &Path, files: &[(&str, &str)]) {
    for (relative, text) in files {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
}

/// Sets or clears read-only on every file and directory under `root`.
fn set_read_only(root: &Path, read_only: bool) {
    let set = |path: &Path| {
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_readonly(read_only);
        fs::set_permissions(path, permissions).unwrap();
    };
    if root.is_dir() {
        // A directory is made writable before its entries, read-only after.
        if !read_only {
            set(root);
        }
        for entry in fs::read_dir(root).unwrap() {
            set_read_only(&entry.unwrap().path(), read_only);
        }
        if read_only {
            set(root);
        }
    } else {
        set(root);
    }
}

/// Every file under `root`, by relative path with `/`, and its text.
fn contents(root: &Path) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let relative = path.strip_prefix(root).unwrap();
                let relative = relative.to_string_lossy().replace('\\', "/");
                found.push((relative, fs::read_to_string(&path).unwrap()));
            }
        }
    }
    found.sort();
    found
}

fn owned(files: &[(&str, &str)]) -> Vec<(String, String)> {
    files
        .iter()
        .map(|(path, text)| (path.to_string(), text.to_string()))
        .collect()
}

/// Read-only source trees under `root`, and the target beside them.
fn sources(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let (first, second) = (root.join("first"), root.join("second"));
    write_tree(&first, &FIRST);
    write_tree(&second, &SECOND);
    set_read_only(&first, true);
    set_read_only(&second, true);
    (first, second, root.join("target"))
}

/// Runs `check`, then removes `root` (read-only parts included) whether
/// or not the check passed.
fn checked(root: &Path, check: impl FnOnce()) {
    let outcome = panic::catch_unwind(AssertUnwindSafe(check));
    set_read_only(root, false);
    fs::remove_dir_all(root).unwrap();
    if let Err(failure) = outcome {
        panic::resume_unwind(failure);
    }
}

/// Whether every file under `root` is writable.
fn all_writable(root: &Path) -> bool {
    contents(root).iter().all(|(relative, _)| {
        !fs::metadata(root.join(relative))
            .unwrap()
            .permissions()
            .readonly()
    })
}

#[test]
fn read_only_trees_sharing_names_export_in_full() {
    let root = scratch("full");
    let (first, second, target) = sources(&root);
    checked(&root, || {
        let done = export(&[&first, &second], &target);
        assert_eq!(done, Ok(Exported { copied: 6, kept: 0 }));
        assert_eq!(contents(&target), owned(&expected()));
        assert!(all_writable(&target), "a copy kept the read-only mode");
        // Done once, it is done.
        let again = export(&[&first, &second], &target);
        assert_eq!(again, Ok(Exported { copied: 0, kept: 6 }));
    });
}

#[test]
fn a_half_done_target_is_completed() {
    let root = scratch("half");
    let (first, second, target) = sources(&root);
    // What the export used to leave: the first tree's files, read-only as
    // their sources, one of them cut short; and a file of the user's.
    write_tree(
        &target,
        &[
            ("a.txt", "first a"),
            ("shared.urdf", "first"),
            ("meshes/x.obj", "x from"),
            ("mine.txt", "the user's own"),
        ],
    );
    for relative in ["a.txt", "shared.urdf", "meshes/x.obj"] {
        set_read_only(&target.join(relative), true);
    }
    checked(&root, || {
        let done = export(&[&first, &second], &target);
        // a.txt was whole; the five others are written.
        assert_eq!(done, Ok(Exported { copied: 5, kept: 1 }));
        let mut wanted = expected();
        wanted.push(("mine.txt", "the user's own"));
        wanted.sort();
        assert_eq!(contents(&target), owned(&wanted));
        // The kept a.txt too.
        assert!(all_writable(&target), "a file was left read-only");
    });
}
