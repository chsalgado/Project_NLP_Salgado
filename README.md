# README

## Prereqs
The following project is meant to be a cross platform executable relying on NET core 3.1 to be built and executed.
You will need to install NET core SDK 3.1 before attempting to build this project.

Manual installation executables can be found [here](https://dotnet.microsoft.com/download/dotnet-core/3.1).

Alternatively, the SDK can be installed by running one of the following 2 scripts: Powershell for Windows and sh for Linux/MacOS.

* [Windows](https://dotnet.microsoft.com/download/dotnet-core/scripts/v1/dotnet-install.ps1)
* [Linux/MacOS](https://dotnet.microsoft.com/download/dotnet-core/scripts/v1/dotnet-install.sh)

Docs to scripts [here](https://docs.microsoft.com/en-us/dotnet/core/tools/dotnet-install-script)

## Expected folder structure
Any given folder containing a ````app.config```` file AND the dataset folder ````Dataset````. In ````Dataset```` folder, additional folders containing the following structure should be placed:  ````Dataset\{corpusName}\CrossVal\{iterationId}\{test\training}```. 

````Dataset\{corpusName}\CrossVal\{iterationId}\```` should contain a ````cats.txt```` file specifying all file ids and their real classification in the format ````{test\training}\{filename} cat1 cat2 ...````. A file ````groundTruth```` similar to ````cats.txt```` but including only test files should also be included.

All files in corpus should be split on ````Dataset\{corpusName}\CrossVal\{iterationId}\{test\training}``` depending on the role they play during the current cross validation iteration.

A sample folder next to the project file (.\Project_NLP_Salgado.csproj\\), named ````Config```` has already been setup for convenience. This folder contains a simple ````app.config```` file with all the experiments used during the project development.

## Building and executing the project
From this ````README```` location (side to side with ````.sln```` file):

````dotnet restore Project_NLP_Salgado.sln````

````dotnet build Project_NLP_Salgado.sln````

````cd Project_NLP_Salgado```` (Navigate to folder containing ````Project_NLP_Salgado.csproj````)

````dotnet run {pathToFolderWithAppConfigAndDataFolders} [-UseSongs]```` ie. ````dotnet run .\Config -UseSongs````. If the use songs is present, songs corpus will be used, else Reuters-21578 will be used.

## Parametizing runs
````app.config```` is the file that setups the multiple runs that will be performed on each project run.

````app.config```` defines one experiment to run per line. Each line is defined by the following spec.

* Empty lines are ignored.
* Lines can be commented out by beggining the line with ````##```` ie. ````## This is a comment````.
* A line is speced as: 

````
-LM {LanguageModelName} [-UnkRatio {unkRatio}] [-SmoothingTechnique {smoothingTechnique}] [-L1 {l1} -L2 {l2} -L3 {l3}] [-IgnoreCase] 
````

## Parameters
* LM: N-Gram Language Model to use. Possible values: ````{Unigram, Bigram, Trigram, TrigramWithLinearInterpolation}````
* SmoothingTechnique: Smoothing technique to use. Possible values: ````{ML, AddK, JM, Dirichlet, AD, TS}```` (Max likelihood, add k, Jelinek-Mercer, Dirichlet, AbsoluteDiscount, Two-stage (Dirichlet + linear interpolation to collection)). ````{JM, Dirichlet, AD, TS}```` are only accurate for Unigram models.
* L1, L2, L3: Smoothing technique hyperparams. For smoothing techniques with only one parameter ````{AddK, JM, Dirichlet, AD}````, only L1 is required. For ````{TS}````, L1 maps to lambda, while L2 to mu. When using ````TrigramWithLinearInterpolation````, all 3 parameters are used.
* IgnoreCase: If present, both training and evaluation sets will be preprossed to lowecase, to trat different casing variations of a word as a single one.


