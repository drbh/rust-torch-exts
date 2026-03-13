{
  description = "dev shell for rust-cuda using host CUDA and pinned LLVM 7";

  inputs = {
    nixpkgs-llvm7.url = "github:NixOS/nixpkgs/nixos-22.11";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs-llvm7, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs-llvm7 {
        inherit system;
        config.allowUnfree = true;
      };
      pkgsModern = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      llvm7 = pkgs.llvmPackages_7.llvm;
      llvm7Dev = pkgs.llvmPackages_7.llvm.dev;
      libclangLib = pkgs.libclang.lib;
      buildWheel = pkgs.writeShellScriptBin "build-wheel" ''
        set -euo pipefail
        rm -f target/wheels/rust_torch_exts-*.whl
        maturin build --release
        echo "==> Wheel is in target/wheels/"
      '';
      buildLocal = pkgs.writeShellScriptBin "build-local" ''
        set -euo pipefail
        rm -rf build/

        # Build the .so via maturin
        maturin build --release
        whl=$(find target/wheels -name 'rust_torch_exts-*.whl' -print -quit)
        tmpdir=$(mktemp -d)
        trap "rm -rf $tmpdir" EXIT
        unzip -q "$whl" -d "$tmpdir"
        so_path=$(find "$tmpdir" -name 'torch_exts*.so' -print -quit)

        # Variant name (matches kernels package convention)
        variant="torch210-cxx11-cu128-$(uname -m)-linux"

        # Detect compute capability
        sm_dot=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')

        out="build/$variant"
        mkdir -p "$out"
        cp "$so_path" "$out/"
        cp python/rust_torch_exts/__init__.py "$out/__init__.py"
        cat > "$out/metadata.json" <<EOF
        {"version":1,"python-depends":[],"backend":{"type":"cuda","archs":["$sm_dot"]}}
        EOF

        echo "==> Built: $out/"
      '';
      # build wheel and kernels compatible
      buildAll = pkgs.writeShellScriptBin "build-all" ''
        set -euo pipefail
        build-wheel
        build-local
      '';
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          bashInteractive
          git
          gnumake
          patchelf
          pkg-config
          rustup
        ] ++ [
          llvm7
          llvm7Dev
        ] ++ [
          pkgsModern.python311
          pkgsModern.maturin
          pkgs.unzip
          buildWheel
          buildLocal
          buildAll
        ];

        # From: https://danieldk.eu/Software/Nix/Nix-CUDA-on-non-NixOS-systems
        #
        # /run/opengl-driver/lib is the standard NixOS path for host GPU drivers.
        # On non-NixOS, set it up once:
        #   sudo mkdir -p /run/opengl-driver/lib
        #   sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so   /run/opengl-driver/lib/
        #   sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /run/opengl-driver/lib/
        # To persist across reboots (systemd):
        #   echo 'L /run/opengl-driver/lib/libcuda.so   - - - - /usr/lib/x86_64-linux-gnu/libcuda.so'   | sudo tee    /etc/tmpfiles.d/nix-opengl-driver.conf
        #   echo 'L /run/opengl-driver/lib/libcuda.so.1 - - - - /usr/lib/x86_64-linux-gnu/libcuda.so.1' | sudo tee -a /etc/tmpfiles.d/nix-opengl-driver.conf
        #   sudo systemd-tmpfiles --create

        shellHook = ''
          export LLVM_CONFIG="${llvm7Dev}/bin/llvm-config"
          export LIBCLANG_PATH="${libclangLib}/lib"

          rust_cuda_dir="$PWD/../rust-cuda"
          rustc_lib_dir="$(rustc --print sysroot)/lib"
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64:$rust_cuda_dir/target/debug:$rustc_lib_dir''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          export LIBRARY_PATH="/usr/local/cuda/lib64''${LIBRARY_PATH:+:$LIBRARY_PATH}"

          if [ ! -e /run/opengl-driver/lib/libcuda.so.1 ]; then
            echo "warning: /run/opengl-driver/lib/libcuda.so.1 not found" >&2
            echo "  see comments in flake.nix for one-time setup instructions" >&2
          fi
        '';
      };
    };
}
