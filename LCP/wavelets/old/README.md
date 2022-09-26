# Istruzioni per usare git con questo progetto

Per prima cosa clonate il repository:
```
git clone https://github.com/zanocrate/LCP_projects_Y4.git
```
Questo creerà una cartella `LCP_projects_Y4` nella directory dove avete eseguito questo comando. 
Dopodichè eseguite questi comandi:
```
cd LCP_projects_Y4
cd .git
nano config
```
E dove c'è `[remote "origin"]`, modificate la voce `url` in modo che includa il vostro Personal Accesso Token, ossia
```
url = https://<TOKEN>@github.com/zanocrate/LCP_projects_Y4.git
```

## Workflow
Prima di mettervi al lavoro e fare qualsiasi cosa, una volta dentro alla cartella `LCP_projects_Y4` fate
```
git fetch origin
git merge
```

Così sarete aggiornati all'ultimo commit che è stato fatto, visto che non lavoriamo su branch diverse per comodità. Ora fate le modifiche che dovete fare, controllate con `git status` cosa aggiungere al commit con `git add` (se fate `git add *` o `git add .` vi aggiungerà automaticamente tutto, tranne i file che decidiamo di ignorare in `.gitignore`), poi fate

```
git commit -m "commento al commit"
git push origin
```
