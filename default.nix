with import <nixpkgs> {}; {
  pyEnv = stdenv.mkDerivation {
    name = "py-321";
    buildInputs = [ 
    	stdenv 
    	python27Full 
    	python27Packages.virtualenv
    	python27Packages.numpy
    	python27Packages.scipy
    	python27Packages.numpy
    	python27Packages.pillow
    	];
    LIBRARY_PATH="${libxml2}/lib";
    shellHook = ''
      # export "PATH=$PATH:`pwd`/venv/bin/"
      # virtualenv --python=python2.7 venv
    '';
  };
}
