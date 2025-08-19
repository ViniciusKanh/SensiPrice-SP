@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1

REM Ir para a raiz do projeto (pasta acima de scripts)
pushd "%~dp0\.."

REM Ativa venv se existir
if exist "env\Scripts\activate.bat" call "env\Scripts\activate.bat"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

REM Se um Excel foi passado como argumento, copia para o caminho esperado
if not "%~1"=="" (
  if not exist "data" mkdir "data"
  copy /Y "%~1" "data\Relacionamento_e_NPS.xlsx" >nul
)

REM Escolhe interpretador Python (python ou py)
set "PY=python"
where %PY% >nul 2>&1 || set "PY=py"

echo === Preparando dataset de sensibilidade a preço ===
%PY% -m train.prep_preco_dataset || goto :err

echo.
echo ✅ Dataset gerado em: data\sensi_preco_dataset.csv
popd
exit /b 0

:err
echo ❌ Erro ao preparar o dataset. Verifique as mensagens acima.
popd
exit /b 1
