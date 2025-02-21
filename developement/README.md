# Developement

- create an issue on github
- In the list of issues, click the issue that you would like to create a branch for.
- In the right sidebar under "Development", click Create a branch. If the issue already has a linked branch or pull request, select  and click Create a branch.
[create branch 1](./img/branch-for-issue_1.png)
[create branch 2](./img/branch-for-issue_2.png)

``` bash
git fetch origin
git checkout 1-prepare-package-from-claudes-code

Depuis github.com:ebraux/sshunet
 * [nouvelle branche] 1-prepare-package-from-claudes-code -> origin/1-prepare-package-from-claudes-code
la branche '1-prepare-package-from-claudes-code' est paramétrée pour suivre 'origin/1-prepare-package-from-claudes-code'.
Basculement sur la nouvelle branche '1-prepare-package-from-claudes-code'

``` 



``` bash
git status
Sur la branche 1-prepare-package-from-claudes-code
Votre branche est à jour avec 'origin/1-prepare-package-from-claudes-code'.

Fichiers non suivis:
  (utilisez "git add <fichier>..." pour inclure dans ce qui sera validé)
	base-code.md
	developement/
	ssh-unet-jupyter.ypnb
	ssh-visualization-functions.py
	unetssh.py
	unetssh2.py
	visualization.md

aucune modification ajoutée à la validation mais des fichiers non suivis sont présents (utilisez "git add" pour les suivre)
``` 

