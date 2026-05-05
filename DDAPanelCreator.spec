# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['dda_panel_creator.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[ 
        'matplotlib', 'scipy',  # The big four bloaters # We do need numpy and pandas
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',    # GUI bloat
        'tkinter', 'test', 'unittest', 'email',    # Standard library bloat
        'notebook', 'IPython',                     # Jupyter bloat
        # More PyQT and other exclusions:
        'qt6', 'qt', 'Qt6', 'Qt', # Binary name variations
        'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
        'tcl', 'tk'
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DDAPanelCreator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DDAPanelCreator',
)
app = BUNDLE(
    coll,
    name='DDAPanelCreator.app',
    icon=None,
    bundle_identifier=None,
    # This fixes a weird bug on macOS where the text would be white on light backgrounds or black on dark backgrounds.
    # This seems to force a theme, fixing this issue.
    info_plist={
        'NSRequiresAquaSystemAppearance': 'True', # Forces Light Mode
        'NSAppearance': 'Aqua',                   # Forces the Aqua (Light) theme
    },
)
