# sekvo


```
cd sekvo/
pyenv install 3.13.1
python -m venv venv_3131/
pip isntall -e '.[all]'


export SEKVO_ANTHROPIC_API_KEY=asdf

# usage
sekvo
sekvo --help
sekvo providers
sekvo --list-commands
sekvo anthropic.generate --help

echo 'tell me a joke' | sekvo anthropic.generate
sekvo anthropic.generate 'tell me a joke'


echo 'tell me a joke' | sekvo anthropic.generate --raw
sekvo anthropic.generate --json 'tell me a joke'

curl https://nebkiso.com > page.txt
echo 'What is this webpage about?\n' > prompt.txt
cat prompt.txt page.txt | sekvo anthropic.generate


```
