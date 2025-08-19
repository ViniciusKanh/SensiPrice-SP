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

echo === Treinando modelo (BERT) para sensibilidade a preço ===
%PY% -m train.treina_preco || goto :err

echo.
echo ✅ Modelo salvo em: app\model\bert-sensi-preco
popd
exit /b 0

:err
echo ❌ Erro no treinamento. Verifique as mensagens acima.
popd
exit /b 1
