### command 1
# add one line at specific line number for given files
# e.g 1: add 'hello' in front of line 4 for 'test.sh'
sed -i '4 s/^/hello\n/' test.sh
# e.g 2: add 'hola" in front of line 4 for all *.yaml
sed -i '4 s/^/hola\n\n/' *.yaml

### command 2
# delete all ".pb" file under one specific directory
find SPECIFIC_DIR -name "*.pb" -exec rm -f {} \;

### command 3
# replace text under specific path
sed -i 's/old_text/new_text/g' /path/to/file

