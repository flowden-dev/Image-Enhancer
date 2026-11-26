@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 goto folder_error
where py >nul 2>nul
if errorlevel 1 goto try_python
for %%V in (3.13 3.14 3.12 3.11) do (
    py -%%V -c "import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYVERSION=%%V"
        goto use_launcher
    )
)
:try_python
where python >nul 2>nul
if not errorlevel 1 goto use_python
echo Python was not found. Install 64-bit Python 3.13 from python.org.
echo Include Tcl/Tk and the Python launcher during installation.
set "RESULT=1"
goto done
:use_launcher
py -%PYVERSION% bootstrap.py --test
set "RESULT=%ERRORLEVEL%"
goto done
:use_python
python bootstrap.py --test
set "RESULT=%ERRORLEVEL%"
goto done
:folder_error
echo Cannot open the extracted tool folder.
pause
exit /b 1
:done
popd
pause
exit /b %RESULT%
