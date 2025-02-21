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



``` bash
git add base-code.md developement/ ssh-unet-jupyter.ypnb ssh-visualization-functions.py unetssh.py unetssh2.py visualization.md 

git status
Sur la branche 1-prepare-package-from-claudes-code
Votre branche est à jour avec 'origin/1-prepare-package-from-claudes-code'.

Modifications qui seront validées :
  (utilisez "git restore --staged <fichier>..." pour désindexer)
	nouveau fichier : base-code.md
	nouveau fichier : developement/README.md
	nouveau fichier : developement/img/branch-for-issue.png
	nouveau fichier : ssh-unet-jupyter.ypnb
	nouveau fichier : ssh-visualization-functions.py
	nouveau fichier : unetssh.py
	nouveau fichier : unetssh2.py
	nouveau fichier : visualization.md
```

``` bash
git commit -m "Initial code update"
[1-prepare-package-from-claudes-code 13f076f] Initial code update
 8 files changed, 2749 insertions(+)
 create mode 100644 base-code.md
 create mode 100644 developement/README.md
 create mode 100644 developement/img/branch-for-issue.png
 create mode 100644 ssh-unet-jupyter.ypnb
 create mode 100644 ssh-visualization-functions.py
 create mode 100644 unetssh.py
 create mode 100644 unetssh2.py
 create mode 100644 visualization.md


git status
Sur la branche 1-prepare-package-from-claudes-code
Votre branche est en avance sur 'origin/1-prepare-package-from-claudes-code' de 1 commit.
  (utilisez "git push" pour publier vos commits locaux)

Modifications qui ne seront pas validées :
  (utilisez "git add <fichier>..." pour mettre à jour ce qui sera validé)
  (utilisez "git restore <fichier>..." pour annuler les modifications dans le répertoire de travail)
	modifié :         developement/README.md

aucune modification n'a été ajoutée à la validation (utilisez "git add" ou "git commit -a")
```

``` bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/ebraux_rsa -o IdentitiesOnly=yes' git push
Énumération des objets: 13, fait.
Décompte des objets: 100% (13/13), fait.
Compression par delta en utilisant jusqu'à 8 fils d'exécution
Compression des objets: 100% (11/11), fait.
Écriture des objets: 100% (12/12), 44.06 Kio | 4.41 Mio/s, fait.
Total 12 (delta 2), réutilisés 0 (delta 0), réutilisés du pack 0
remote: Resolving deltas: 100% (2/2), done.
To github.com:ebraux/sshunet.git
   842fc57..13f076f  1-prepare-package-from-claudes-code -> 1-prepare-package-from-claudes-code

git status
Sur la branche 1-prepare-package-from-claudes-code
Votre branche est à jour avec 'origin/1-prepare-package-from-claudes-code'.

Modifications qui ne seront pas validées :
  (utilisez "git add <fichier>..." pour mettre à jour ce qui sera validé)
  (utilisez "git restore <fichier>..." pour annuler les modifications dans le répertoire de travail)
	modifié :         developement/README.md

aucune modification n'a été ajoutée à la validation (utilisez "git add" ou "git commit -a")
```



