@echo off

REM 检查项目内是否已安装Miniconda，若未安装则下载并安装
set MINICONDA_DIR=%~dp0miniconda
if not exist "%MINICONDA_DIR%\Scripts\conda.exe" (
   echo Downloading Miniconda...
   powershell -Command "Invoke-WebRequest https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile miniconda.exe"
   echo Installing Miniconda to %MINICONDA_DIR%...
   start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_DIR%
)

REM 添加Miniconda到系统路径（临时修改当前会话的PATH）
set "PATH=%MINICONDA_DIR%;%MINICONDA_DIR%\Scripts;%PATH%"

REM 创建项目环境并安装依赖
@REM call %MINICONDA_DIR%\Scripts\conda.exe init
@REM call %MINICONDA_DIR%\Scripts\conda.exe env create -f ASR_trans.yaml --name ASR_trans
call activate ASR_trans

REM （可选）测试环境是否正确设置
REM python -c "import some_dependency; print(some_dependency.__version__)"

REM 运行 main.py
python main.py %*

REM Deactivate the environment (optional)
REM call C:\miniconda\Scripts\conda.exe deactivate

PAUSE