with import <nixpkgs> {}; {
  pyEnv = stdenv.mkDerivation {
    name = "py";
    buildInputs = [ stdenv python27Full python27Packages.virtualenv libxml2 ];
    LIBRARY_PATH="${libxml2}/lib";
    shellHook = ''
      virtualenv --python=python2.7 venv
    '';
  };
}
