@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1

pushd "%~dp0\.."

REM Ativa venv se existir
if exist "env\Scripts\activate.bat" call "env\Scripts\activate.bat"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

set "PY=python"
where %PY% >nul 2>&1 || set "PY=py"

REM Defaults
set "ARQ=data\Relacionamento_e_NPS.xlsx"
set "COL=Resumo"
set "OUT="
set "MDIR=app\model\bert-sensi-preco"

REM Sobrescreve pelos argumentos, se fornecidos
if not "%~1"=="" set "ARQ=%~1"
if not "%~2"=="" set "COL=%~2"
if not "%~3"=="" set "OUT=%~3"
if not "%~4"=="" set "MDIR=%~4"

REM Monta comando
set "CMD=%PY% -m app.inferir_preco_excel --arquivo \"%ARQ%\" --coluna \"%COL%\" --modelo \"%MDIR%\""
if not "%OUT%"=="" set "CMD=%CMD% --saida \"%OUT%\""

echo === Inferindo sensibilidade a preço ===
echo Arquivo..: %ARQ%
echo Coluna...: %COL%
echo Modelo...: %MDIR%
if not "%OUT%"=="" echo Saída....: %OUT%

%CMD% || goto :err

echo.
echo ✅ Inferência concluída.
popd
exit /b 0

:err
echo ❌ Erro na inferência. Verifique as mensagens acima.
popd
exit /b 1
