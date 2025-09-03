#!/bin/bash
rsync --inplace --whole-file --progress -ahe ssh  --exclude=".*" --exclude="**/__pycache__" --no-links --update ./* ubelix:/storage/homefs/dn25y590/projects/medgemma-chat
