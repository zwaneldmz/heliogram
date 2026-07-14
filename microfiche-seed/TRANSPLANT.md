# Transplanting this seed into the real repo

This directory is the initial content of the `microfiche` repository, parked here because
the session's GitHub integration cannot create repositories.

1. Create an empty repo at github.com/new (name: `microfiche`, no README/license — or
   with them; the license should be Apache-2.0).
2. From a clone of heliogram at this branch:

```sh
cp -r microfiche-seed /tmp/microfiche && cd /tmp/microfiche
rm TRANSPLANT.md
git init -b main && git add -A && git commit -m "Seed: concept README and build plan"
git remote add origin git@github.com:zwaneldmz/microfiche.git
git push -u origin main
```

3. Delete `microfiche-seed/` from heliogram afterwards.
