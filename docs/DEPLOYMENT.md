# 🚀 Deploy em Produção - CeciAI

**Guia completo para colocar CeciAI em produção.**

> **Nota:** Para instalação local/desenvolvimento, veja [INSTALL.md](../INSTALL.md)

---

## 📋 Índice

1. [Opções de Deploy](#-opções-de-deploy)
2. [Deploy com Docker](#-deploy-com-docker)
3. [Deploy Linux Servidor](#-deploy-linux-servidor)
4. [Configuração de Produção](#-configuração-de-produção)
5. [Segurança](#-segurança)
6. [Monitoramento](#-monitoramento)
7. [Deploy em Cloud](#-deploy-em-cloud)

---

## 🎯 Opções de Deploy

| Opção | Uso | Recomendado Para |
|-------|-----|------------------|
| **Docker** 🐳 | Containers | Produção pequena/média |
| **Linux Servidor** 🖥️ | Instalação nativa | Produção alta escala |
| **Cloud (AWS/GCP)** ☁️ | Infraestrutura gerenciada | Produção profissional |

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

## 🖥️ Deploy Linux Servidor

### Instalação em Servidor Linux

> **Pré-requisito:** Ter Python, Git e Ollama instalados (veja [INSTALL.md](../INSTALL.md))

```bash
# 1. Criar usuário para CeciAI
sudo useradd -r -m -s /bin/bash ceciai

# 2. Clonar projeto para /opt
sudo git clone https://github.com/lukeware-digital/ai-invest.git /opt/ceci-ai
sudo chown -R ceciai:ceciai /opt/ceci-ai

# 3. Instalar como usuário ceciai
sudo -u ceciai bash << 'EOF'
cd /opt/ceci-ai
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
EOF

# 4. Configurar ambiente de produção
sudo -u ceciai cp /opt/ceci-ai/.env.example /opt/ceci-ai/.env
# Editar /opt/ceci-ai/.env com configurações de produção

# 5. Instalar Nginx e Supervisor
sudo apt install nginx supervisor -y  # Ubuntu/Debian
# sudo dnf install nginx supervisor -y  # CentOS/RHEL
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

---

## ✅ Checklist de Deploy em Produção

### Infraestrutura
- [ ] Servidor com 8GB+ RAM, 4+ cores
- [ ] Python 3.11+ instalado
- [ ] Git, Ollama instalados
- [ ] Firewall configurado (UFW)
- [ ] SSL/TLS configurado (Let's Encrypt)

### Aplicação
- [ ] Código em `/opt/ceci-ai`
- [ ] Usuário `ceciai` criado
- [ ] `.env` configurado para produção
- [ ] Serviços systemd (API + Ollama) criados
- [ ] Nginx configurado como proxy

### Segurança
- [ ] Firewall ativo (80, 443, SSH)
- [ ] SSL funcionando
- [ ] Logs sendo gravados

### Validação Final
- [ ] `curl https://seu-dominio.com/health` retorna 200
- [ ] Dashboard acessível
- [ ] Análise funciona sem erros

---

**🎉 Produção pronta!**

**Documentos relacionados:**
- 📦 [INSTALL.md](../INSTALL.md) - Instalação local/desenvolvimento
- 🚀 [QUICK_START.md](QUICK_START.md) - Como usar o sistema
