# Tutorial Deepracer-for-cloud


## 1 - Conexão com servidor

Obs.: Se a solução está sendo executada localmente, essa seção pode ser desconsiderada.

Obs2.: O Inteli possui um servidor chamado Shrek2 e ele será o alvo da nossa conexão. Sendo assim, as instruções indicadas funcionaram apenas nesse host.

Para iniciar a conexão com o servidor, existe a necessidade de que o sistema local possui o OpenSSH instalado. Devido o fato de que esse software é bastante comum nas distribuições dos sistemas operacionais mais famosos, o seu guia de instalação não será contemplado nesse documento. Para mais informações, acesse: [https://www.openssh.com/](https://www.openssh.com/)

Com o OpenSSH instalado, a conexão será iniciada com o seguinte comando: `ssh deepracer@shrek2.inteli.local`. Esse comando deve ser executado em um emulador de terminal e a máquina local precisará estar dento da rede interna da faculdade. Logo após isso, o programa irá solicitar senha do usuário "deepracer". Por fins de segurança, essa senha também não será fornecida nesse documento.

Por fim, a interface do terminal do servidor estará visível no emulador de terminal e ela será utilizada para rodar os comandos nas seções seguintes.

## 2 - Iniciando o Deepracer-for-cloud usando a pista padrão


Para iniciar o deepracer-for-cloud na pista padrão, é necessário ter como parâmetro um número de um grupo. Cada grupo possui a sua própria série de arquivos para realizar o seu treinamento. Como exemplo, os arquivos do grupo 3 serão utilizados.

Como cada grupo possui um nome de arquivo diferente e o deepracer-for-cloud possui uma nomeclatura padrão, a criação de soft links será necessária para dar prosseguimento com o treino. O comando para fazer esses links é: `cd ~/deepracer-for-cloud && ln -s run3.env run.env && ln -s system3.env system.env && ln -s custom_files3 custom_files`. O "3" simboliza o número do grupo que está sendo utilizado.

Com os links devidamente criados, o próximo passo é executar o script "activate.sh" no shell que está sendo utilizado. Para isso, deve-se rodar o comando `cd ~/deepracer-for-cloud && source bin/activate.sh`, isso fará com que os próximos comandos de atalho sejam reconhecidos pelo shell.

Com o shell apropriadamente ajustado. As configurações do ambiente devem ser atualizadas para que o treinamento ocorra da maneira adequada. Para isso, o comando `cd ~/deepracer-for-cloud && dr-update && dr-update-env && dr-upload-custom-files` deve ser executado.

Depois da atualização dos arquivos, a única instrução restante é o `dr-start-training -w`, que iniciará um treino do zero na sessão de terminal que está sendo utilizada.

Obs3.: Rodar o comando de inicio de treino com a flag '-w' irá reiniciar todo o aprendizado do carrinho. Antes de executá-lo, é recomendado que todos os arquivos de resultado (cobertos em uma seção posterior do documento) sejam devidamente armazenados em outro local.

Obs4.: Para visualizar o treinamento em uma interface gráfica, uma ponte deve ser feita durante a conexão SSH. Para isso, é necessário voltar para uma sessão de terminal na máquina host, 
