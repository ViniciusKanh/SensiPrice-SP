@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1

REM Vai pra raiz do projeto
pushd "%~dp0\.."

REM Ativa venv se existir
if exist "env\Scripts\activate.bat" call "env\Scripts\activate.bat"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

set "PY=python"
where %PY% >nul 2>&1 || set "PY=py"

echo === Atualizando requirements.txt (freeze do ambiente atual) ===
REM PowerShell é usado aqui pra filtrar linhas indesejadas
powershell -NoProfile -Command ^
  "& { %PY% -m pip freeze ^| Select-String -NotMatch '^(pip==|setuptools==|wheel==)$' ^| Set-Content -Encoding UTF8 'requirements.txt' }" ^
  || goto :err

echo ✅ requirements.txt atualizado.
popd
exit /b 0

:err
echo ❌ Falhou ao atualizar requirements.txt.
popd
exit /b 1
