# 🚀 CeciAI - Guia de Deploy

**Deploy completo em produção**

---

## 🎯 Opções de Deploy

### 1. **Docker (Recomendado)** 🐳
- ✅ Mais fácil e rápido
- ✅ Isolamento completo
- ✅ Funciona em qualquer sistema

### 2. **WSL2 Ubuntu** 🐧
- ✅ Performance nativa Linux
- ✅ Melhor para desenvolvimento
- ✅ Acesso direto ao hardware

### 3. **Linux Nativo** 🖥️
- ✅ Máxima performance
- ✅ Para servidores dedicados
- ✅ Produção de alta escala

---

## 🐳 Deploy com Docker

### Instalação Rápida (1 comando)

```bash
# Clonar e fazer build completo
git clone <repo-url>
cd ceci-ai
make build

# Iniciar sistema
make up

# Verificar se está funcionando
curl http://localhost:8000/health
```

### Comandos Docker Úteis

```bash
# Build e deploy
make build          # Build completo (inclui Ollama)
make up             # Iniciar containers
make down           # Parar containers
make restart        # Reiniciar containers
make logs           # Ver logs em tempo real
make clean          # Limpar tudo

# Comandos individuais
docker-compose up -d                    # Iniciar em background
docker-compose logs -f ceciai-api       # Logs da API
docker-compose exec ceciai-api bash     # Entrar no container
docker-compose ps                       # Status dos containers
```

### Configuração Docker

**docker-compose.yml** (Desenvolvimento):
```yaml
version: '3.8'

services:
  ceciai-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CECIAI_ENV=development
      - OLLAMA_HOST=http://ollama:11434
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_MODELS=llama3.2:3b

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  ollama_data:
  redis_data:
```

**docker-compose.prod.yml** (Produção):
```yaml
version: '3.8'

services:
  ceciai-api:
    build: .
    ports:
      - "80:8000"
    environment:
      - CECIAI_ENV=production
      - OLLAMA_HOST=http://ollama:11434
      - ENABLE_TRADING=true
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - ceciai-api
    restart: unless-stopped

volumes:
  ollama_data:
  redis_data:
```

---

## 🐧 Deploy no WSL2 Ubuntu

### 1. Preparar WSL2

```bash
# No PowerShell (Windows)
wsl --install Ubuntu-24.04
wsl --set-default Ubuntu-24.04

# Configurar recursos (.wslconfig)
# C:\Users\<user>\.wslconfig
[wsl2]
memory=24GB
processors=14
swap=8GB
gpuSupport=true
localhostForwarding=true
```

### 2. Instalar Dependências

```bash
# Entrar no WSL2
wsl

# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Python 3.12
sudo apt install python3.12 python3.12-venv python3-pip -y

# Ferramentas de desenvolvimento
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    software-properties-common

# CUDA (para GPU)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4 -y

# Adicionar ao PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Instalar CeciAI

```bash
# Clonar projeto
cd ~
mkdir -p workspace
cd workspace
git clone <repo-url> ceci-ai
cd ceci-ai

# Ambiente Python
python3 -m venv venv
source venv/bin/activate

# TA-Lib (análise técnica)
sudo apt install -y libta-lib0-dev ta-lib
pip install TA-Lib

# Dependências do projeto
pip install -r requirements.txt

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:3b

# Configurar ambiente
cp .env.example .env
# Editar .env conforme necessário

# Testar sistema
python scripts/test_complete_system.py
```

### 4. Configurar Serviços

```bash
# Criar serviço systemd para CeciAI
sudo tee /etc/systemd/system/ceciai.service << EOF
[Unit]
Description=CeciAI Trading System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/workspace/ceci-ai
Environment=PATH=/home/$USER/workspace/ceci-ai/venv/bin
ExecStart=/home/$USER/workspace/ceci-ai/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Habilitar e iniciar
sudo systemctl daemon-reload
sudo systemctl enable ceciai
sudo systemctl start ceciai

# Verificar status
sudo systemctl status ceciai
```

### 5. Scripts de Inicialização

```bash
# Criar script de startup
cat > ~/start_ceciai.sh << 'EOF'
#!/bin/bash

echo "🚀 Iniciando CeciAI..."

# Ativar venv
cd ~/workspace/ceci-ai
source venv/bin/activate

# Verificar Ollama
if ! pgrep -x "ollama" > /dev/null; then
    echo "Iniciando Ollama..."
    ollama serve &
    sleep 5
fi

# Verificar GPU
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU OK: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    echo "⚠️  GPU não detectada"
fi

# Iniciar API
echo "Iniciando API..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

echo "✅ CeciAI iniciado!"
echo "API: http://localhost:8000"
echo "Health: http://localhost:8000/health"
EOF

chmod +x ~/start_ceciai.sh
```

---

## 🖥️ Deploy Linux Nativo

### Ubuntu/Debian

```bash
# 1. Atualizar sistema
sudo apt update && sudo apt upgrade -y

# 2. Python 3.12
sudo apt install python3.12 python3.12-venv python3-pip -y

# 3. Dependências
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    libta-lib0-dev \
    ta-lib \
    nginx \
    supervisor

# 4. Clonar e configurar projeto
git clone <repo-url> /opt/ceci-ai
cd /opt/ceci-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:3b

# 6. Configurar
cp .env.example .env
# Editar .env para produção

# 7. Testar
python scripts/test_complete_system.py
```

### CentOS/RHEL

```bash
# 1. Atualizar sistema
sudo dnf update -y

# 2. Python 3.12
sudo dnf install python3.12 python3.12-pip python3.12-devel -y

# 3. Dependências
sudo dnf groupinstall "Development Tools" -y
sudo dnf install git curl wget nginx supervisor -y

# 4. TA-Lib (compilar do source)
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..

# 5. Continuar com passos similares ao Ubuntu
```

---

## ⚙️ Configuração de Produção

### Nginx (Proxy Reverso)

```nginx
# /etc/nginx/sites-available/ceciai
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/ceciai.crt;
    ssl_certificate_key /etc/ssl/private/ceciai.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # API Proxy
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long analysis
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/ceci-ai/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
}
```

### Supervisor (Gerenciamento de Processos)

```ini
# /etc/supervisor/conf.d/ceciai.conf
[program:ceciai-api]
command=/opt/ceci-ai/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 4
directory=/opt/ceci-ai
user=ceciai
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/ceciai/api.log
environment=PATH="/opt/ceci-ai/venv/bin"

[program:ollama]
command=/usr/local/bin/ollama serve
user=ceciai
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/ceciai/ollama.log
```

### Systemd Services

```ini
# /etc/systemd/system/ceciai-api.service
[Unit]
Description=CeciAI API Server
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=ceciai
Group=ceciai
WorkingDirectory=/opt/ceci-ai
Environment=PATH=/opt/ceci-ai/venv/bin
ExecStart=/opt/ceci-ai/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/ollama.service
[Unit]
Description=Ollama LLM Server
After=network.target

[Service]
Type=simple
User=ceciai
Group=ceciai
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=10
Environment=OLLAMA_HOST=127.0.0.1:11434

[Install]
WantedBy=multi-user.target
```

---

## 🔒 Segurança

### Firewall (UFW)

```bash
# Configurar firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Permitir apenas portas necessárias
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Verificar status
sudo ufw status
```

### SSL/TLS (Let's Encrypt)

```bash
# Instalar Certbot
sudo apt install certbot python3-certbot-nginx -y

# Obter certificado
sudo certbot --nginx -d your-domain.com

# Auto-renovação
sudo crontab -e
# Adicionar linha:
# 0 12 * * * /usr/bin/certbot renew --quiet
```

### Usuário Dedicado

```bash
# Criar usuário para CeciAI
sudo useradd -r -s /bin/false ceciai
sudo mkdir -p /opt/ceci-ai
sudo chown -R ceciai:ceciai /opt/ceci-ai

# Configurar permissões
sudo chmod 750 /opt/ceci-ai
sudo chmod -R 640 /opt/ceci-ai/.env
```

---

## 📊 Monitoramento

### Logs

```bash
# Logs da API
tail -f /var/log/ceciai/api.log

# Logs do Ollama
tail -f /var/log/ceciai/ollama.log

# Logs do sistema
journalctl -u ceciai-api -f
journalctl -u ollama -f

# Logs do Nginx
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### Health Checks

```bash
# Script de monitoramento
cat > /opt/ceci-ai/scripts/health_check.sh << 'EOF'
#!/bin/bash

API_URL="http://localhost:8000/health"
OLLAMA_URL="http://localhost:11434/api/tags"

# Check API
if curl -s $API_URL | grep -q "healthy"; then
    echo "✅ API: OK"
else
    echo "❌ API: FAIL"
    # Restart API service
    sudo systemctl restart ceciai-api
fi

# Check Ollama
if curl -s $OLLAMA_URL | grep -q "models"; then
    echo "✅ Ollama: OK"
else
    echo "❌ Ollama: FAIL"
    # Restart Ollama service
    sudo systemctl restart ollama
fi
EOF

chmod +x /opt/ceci-ai/scripts/health_check.sh

# Adicionar ao cron (verificar a cada 5 minutos)
echo "*/5 * * * * /opt/ceci-ai/scripts/health_check.sh >> /var/log/ceciai/health.log 2>&1" | sudo crontab -
```

### Métricas de Sistema

```bash
# Instalar htop e iotop
sudo apt install htop iotop -y

# Monitorar recursos
htop           # CPU e RAM
iotop          # Disk I/O
nvidia-smi     # GPU (se disponível)

# Disk usage
df -h
du -sh /opt/ceci-ai/data/
```

---

## 🚀 Deploy em Cloud

### AWS EC2

```bash
# 1. Criar instância EC2
# - Tipo: t3.large ou maior
# - OS: Ubuntu 24.04 LTS
# - Storage: 50GB+ SSD
# - Security Group: HTTP (80), HTTPS (443), SSH (22)

# 2. Conectar via SSH
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Seguir passos de instalação Linux nativo
# 4. Configurar Elastic IP
# 5. Configurar Route 53 para DNS
```

### Google Cloud Platform

```bash
# 1. Criar VM instance
gcloud compute instances create ceciai-vm \
    --image-family=ubuntu-2404-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=e2-standard-4 \
    --boot-disk-size=50GB \
    --tags=http-server,https-server

# 2. Configurar firewall
gcloud compute firewall-rules create allow-ceciai \
    --allow tcp:80,tcp:443 \
    --target-tags http-server,https-server

# 3. SSH e instalar
gcloud compute ssh ceciai-vm
# Seguir passos de instalação
```

### DigitalOcean

```bash
# 1. Criar Droplet
# - Ubuntu 24.04 x64
# - 4GB RAM, 2 vCPUs
# - 50GB SSD

# 2. SSH
ssh root@your-droplet-ip

# 3. Seguir instalação Linux nativo
```

---

## 🔧 Troubleshooting

### API não inicia

```bash
# Verificar logs
journalctl -u ceciai-api -n 50

# Verificar porta
sudo lsof -i :8000

# Testar manualmente
cd /opt/ceci-ai
source venv/bin/activate
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Ollama não responde

```bash
# Verificar processo
ps aux | grep ollama

# Reiniciar
sudo systemctl restart ollama

# Testar conexão
curl http://localhost:11434/api/tags
```

### GPU não detectada

```bash
# Verificar driver NVIDIA
nvidia-smi

# Verificar CUDA
nvcc --version

# Reinstalar driver se necessário
sudo apt purge nvidia-*
sudo apt install nvidia-driver-535
sudo reboot
```

### Performance baixa

```bash
# Verificar recursos
htop
free -h
df -h

# Verificar swap
swapon --show

# Otimizar se necessário
sudo sysctl vm.swappiness=10
```

---

## ✅ Checklist de Deploy

### Pré-deploy
- [ ] Hardware compatível (8GB+ RAM, 4+ cores)
- [ ] Sistema operacional atualizado
- [ ] Dependências instaladas
- [ ] GPU configurada (se disponível)

### Deploy
- [ ] Código clonado e configurado
- [ ] Ambiente virtual criado
- [ ] Dependências Python instaladas
- [ ] Ollama instalado e modelos baixados
- [ ] Arquivo .env configurado
- [ ] Testes passando

### Produção
- [ ] Nginx configurado
- [ ] SSL/TLS configurado
- [ ] Firewall configurado
- [ ] Serviços systemd criados
- [ ] Logs configurados
- [ ] Monitoramento ativo
- [ ] Backups configurados

### Validação
- [ ] Health check retorna 200 OK
- [ ] API responde em < 60s
- [ ] Análise completa funciona
- [ ] Logs sem erros críticos
- [ ] Recursos dentro dos limites

---

**🎉 Deploy completo! Sistema pronto para produção!** 🚀

**Próximos passos:**
1. Configurar monitoramento avançado
2. Implementar alertas
3. Configurar backups automáticos
4. Testar com capital real (pequeno)
5. Escalar conforme necessário

**Versão:** 1.0.0  
**Última atualização:** 2025-10-08
