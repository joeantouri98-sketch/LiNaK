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
modules = Core,DBus,Gui,Network,OpenGL,Positioning,PrintSupport,Qml,QmlMeta,QmlModels,QmlWorkerScript,Quick,QuickWidgets,WebChannel,WebEngineCore,WebEngineWidgets,Widgets
plugins = accessiblebridge,egldeviceintegrations,generic,iconengines,imageformats,networkaccess,networkinformation,platforminputcontexts,platforms,platforms/darwin,platformthemes,position,printsupport,qmltooling,scenegraph,styles,tls,vectorimageformats,wayland-decoration-client,wayland-graphics-integration-client,wayland-shell-integration,xcbglintegrations

[nuitka]
extra_args = --quiet --noinclude-qt-translations --include-package=linak --include-data-dir=resources=resources
macos.permissions = 
mode = standalone

