use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn lib_exists(dir: &Path, name: &str) -> bool {
    // Windows static: Foo.lib；Unix static: libFoo.a
    let win = dir.join(format!("{name}.lib"));
    let unix = dir.join(format!("lib{name}.a"));
    win.exists() || unix.exists()
}

fn main() {
    println!("cargo:rerun-if-env-changed=BULLET_SKIP_ASSET_EXPORT");
    let mut cfg = cmake::Config::new("bullet3");
    cfg.profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_BULLET2_DEMOS", "ON")
        .define("BUILD_CPU_DEMOS", "OFF")
        .define("BUILD_OPENGL3_DEMOS", "ON")
        .define("BUILD_EXTRAS", "ON")
        .define("BUILD_UNIT_TESTS", "OFF")
        .define("BUILD_PYBULLET", "OFF");

    // cfg.generator("Ninja");

    // cfg.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded$<$<CONFIG:Debug>:Debug>");

    let dst = cfg.build();
    let mut candidates: Vec<PathBuf> = vec![dst.join("lib")];
    candidates.push(dst.join("build").join("lib"));
    candidates.push(dst.join("build").join("lib").join("Release"));
    candidates.push(dst.join("build").join("lib").join("RelWithDebInfo"));
    candidates.push(dst.join("build").join("lib").join("Debug"));

    let libdir = candidates
        .into_iter()
        .find(|dir| lib_exists(dir, "LinearMath"))
        .unwrap_or_else(|| {
            panic!(
                "failed to locate built Bullet libraries under {}",
                dst.display()
            )
        });

    println!("cargo:rustc-link-search=native={}", libdir.display());

    for &lib in &[
        "BulletSoftBody",
        "BulletDynamics",
        "BulletCollision",
        "LinearMath",
        "BulletInverseDynamicsUtils",
        "BulletInverseDynamics",
        "Bullet3Common",
        "Bullet3Collision",
        "Bullet3Dynamics",
        "Bullet3Geometry",
        "BulletFileLoader",
        "BulletWorldImporter",
        "BulletExampleBrowserLib",
        "BulletRoboticsGUI",
        "gwen",
        "OpenGLWindow",
        "Bullet3AppSupport",
        "BulletRobotics",
        "Bullet3OpenCL_clew",
        "cbullet",
    ] {
        if lib_exists(&libdir, lib) {
            println!("cargo:rustc-link-lib=static={lib}");
        }
    }

    #[cfg(all(not(target_env = "msvc"), not(target_os = "macos")))]
    println!("cargo:rustc-link-lib=dylib=stdc++");

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=dylib=User32");
        println!("cargo:rustc-link-lib=dylib=Gdi32");
        println!("cargo:rustc-link-lib=dylib=Opengl32");
        println!("cargo:rustc-link-lib=dylib=Comdlg32");
    }

    if env::var_os("BULLET_SKIP_ASSET_EXPORT").as_deref() != Some(std::ffi::OsStr::new("1")) {
        if let Err(err) = export_bullet_data() {
            println!("cargo:warning=failed to prepare bullet data directory: {err}");
        }
    }
}

fn export_bullet_data() -> Result<(), String> {
    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").map_err(|e| format!("missing CARGO_MANIFEST_DIR: {e}"))?;
    let source_1 = Path::new(&manifest_dir)
        .join("bullet3")
        .join("examples")
        .join("pybullet")
        .join("gym")
        .join("pybullet_data");
    if !source_1.exists() {
        return Err(format!(
            "source data directory not found: {}",
            source_1.display()
        ));
    }
    let source_2 = Path::new(&manifest_dir).join("bullet3").join("data");
    if !source_2.exists() {
        return Err(format!(
            "source data directory not found: {}",
            source_2.display()
        ));
    }

    let target_root =
        bullet_data_target_dir().ok_or_else(|| "could not determine user directory".to_string())?;
    let target = target_root.join("bullet");

    // Builds must never delete or overwrite an existing user asset directory.
    // Users can point examples at a separate freshly exported data directory
    // when an asset refresh is needed.
    if target.symlink_metadata().is_ok() {
        println!(
            "cargo:warning=preserving existing Bullet asset path {}; set BULLET_SKIP_ASSET_EXPORT=1 to disable asset export",
            target.display()
        );
        return Ok(());
    }
    copy_dir_recursive(&source_1, &target)?;
    copy_dir_recursive(&source_2, &target)?;

    Ok(())
}

fn bullet_data_target_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
    }

    #[cfg(target_os = "macos")]
    {
        env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join("Library").join("Application Support"))
    }

    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".local").join("share"))
    }
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<(), String> {
    fs::create_dir_all(destination)
        .map_err(|e| format!("failed to create {}: {e}", destination.display()))?;

    for entry in
        fs::read_dir(source).map_err(|e| format!("failed to read {}: {e}", source.display()))?
    {
        let entry = entry.map_err(|e| format!("failed to iterate directory: {e}"))?;
        let path = entry.path();
        let dest_path = destination.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &dest_path)?;
        } else {
            fs::copy(&path, &dest_path).map_err(|e| {
                format!(
                    "failed to copy {} to {}: {e}",
                    path.display(),
                    dest_path.display()
                )
            })?;
        }
    }

    Ok(())
}
