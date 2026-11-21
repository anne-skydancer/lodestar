use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=LODESTAR_LIB_DIR");
    let target = env::var("CARGO_CFG_TARGET_OS").expect("Cargo target OS is required");
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let library = match (target.as_str(), target_env.as_str()) {
        ("windows", "msvc") => "lodestar_c.lib",
        ("windows", _) => "liblodestar_c.dll.a",
        ("macos", _) => "liblodestar_c.dylib",
        _ => "liblodestar_c.so",
    };
    let dir = if let Some(dir) = env::var_os("LODESTAR_LIB_DIR") {
        PathBuf::from(dir)
    } else {
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        ["build/Release", "build", "build-ninja"]
            .into_iter()
            .map(|path| repo.join(path))
            .find(|path| path.join(library).is_file())
            .expect("Build the native lodestar_c target and set LODESTAR_LIB_DIR to its output directory")
    };
    let dir = dir.canonicalize().unwrap_or_else(|error| {
        panic!(
            "Unable to open Lodestar directory {} ({error})",
            dir.display()
        );
    });
    let link_file = dir.join(library);
    assert!(
        link_file.is_file(),
        "Missing native library {}",
        link_file.display()
    );
    println!("cargo:rerun-if-changed={}", link_file.display());
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=dylib=lodestar_c");
}
