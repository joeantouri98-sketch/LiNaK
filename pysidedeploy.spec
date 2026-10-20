[app]
title = LiNaK
project_dir = .
input_file = src/linak/__main__.py
exec_directory = dist/pyside6
icon = LiNaK.png

[python]
packages = Nuitka==4.1.1

[qt]
qml_files = 
excluded_qml_plugins = 
modules = Core,Gui,WebEngineWidgets,Widgets
plugins = accessiblebridge,egldeviceintegrations,generic,iconengines,imageformats,networkaccess,networkinformation,platforminputcontexts,platforms,platforms/darwin,platformthemes,position,printsupport,qmltooling,scenegraph,styles,tls,vectorimageformats,wayland-decoration-client,wayland-graphics-integration-client,wayland-shell-integration,xcbglintegrations

[nuitka]
extra_args = --quiet --noinclude-qt-translations --include-package=linak --include-package=plotly --noinclude-custom-mode=plotly:bytecode --include-package-data=plotly:package_data/templates/*.json --include-package-data=plotly:validators/_validators.json --include-distribution-metadata=plotly --noinclude-data-files=plotly/package_data/plotly.min.js --noinclude-data-files=plotly/package_data/datasets/* --noinclude-data-files=plotly/labextension/** --include-data-dir=src/linak/resources=linak/resources --include-data-dir=src/linak/report_templates=linak/report_templates --disable-console --force-stdout-spec={TEMP}/LiNaK_stdout_{PID}.log --force-stderr-spec={TEMP}/LiNaK_stderr_{PID}.log --assume-yes-for-downloads
macos.permissions = 
mode = onefile

