with import <nixpkgs> {}; {
  pyEnv = stdenv.mkDerivation {
    name = "py-321";
    buildInputs = [ 
      # python and packages
    	stdenv 
    	python27Full 
    	python27Packages.numpy
    	python27Packages.scipy
    	python27Packages.numpy
    	python27Packages.pillow
      python27Packages.matplotlib
      pygtk

      # report
      pandoc
      (pkgs.texLiveAggregationFun{
        paths = [
          texLive
          texLiveExtra
          lmodern
        ];})
      fswatch
    ];
    LIBRARY_PATH="${libxml2}/lib";
    shellHook = ''
      # export "PATH=$PATH:`pwd`/venv/bin/"
      # virtualenv --python=python2.7 venv
    '';
  };
}
