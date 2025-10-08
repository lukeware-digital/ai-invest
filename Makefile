# ==========================================
# CeciAI - Makefile
# Comandos para desenvolvimento e produção
# ==========================================

.PHONY: help build install clean test

# Variáveis
DOCKER_COMPOSE = /tmp/docker-compose
DOCKER = docker
PYTHON = python3.11
PROJECT_NAME = ceciai
IMAGE_NAME = ceciai:latest

# Cores para output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
BLUE = \033[0;34m
NC = \033[0m # No Color

# ==================== HELP ====================

help: ## Mostra esta ajuda
	@echo "$(GREEN)╔══════════════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║                                                      ║$(NC)"
	@echo "$(GREEN)║     CeciAI - Comandos Disponíveis                   ║$(NC)"
	@echo "$(GREEN)║                                                      ║$(NC)"
	@echo "$(GREEN)╚══════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# ==================== BUILD COMPLETO ====================

build: ## 🚀 Build completo: instala tudo, configura e sobe aplicação
	@echo "$(GREEN)╔══════════════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║                                                      ║$(NC)"
	@echo "$(GREEN)║     🚀 CeciAI - Build Completo                      ║$(NC)"
	@echo "$(GREEN)║                                                      ║$(NC)"
	@echo "$(GREEN)╚══════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@$(MAKE) check-system
	@$(MAKE) setup-env
	@$(MAKE) install-ollama
	@$(MAKE) validate-ollama
	@$(MAKE) docker-build
	@$(MAKE) docker-up
	@$(MAKE) verify-deployment
	@echo ""
	@echo "$(GREEN)╔══════════════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║                                                      ║$(NC)"
	@echo "$(GREEN)║     ✅ Build Concluído com Sucesso!                 ║$(NC)"
	@echo "$(GREEN)║                                                      ║$(NC)"
	@echo "$(GREEN)╚══════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@$(MAKE) show-info

# ==================== VERIFICAÇÕES ====================

check-system: ## Verifica requisitos do sistema
	@echo "$(BLUE)🔍 Verificando sistema...$(NC)"
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)❌ Docker não encontrado. Instale: https://docs.docker.com/get-docker/$(NC)"; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "$(RED)❌ Docker Compose não encontrado$(NC)"; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "$(RED)❌ Python 3 não encontrado$(NC)"; exit 1; }
	@echo "$(GREEN)✅ Sistema OK$(NC)"

# ==================== CONFIGURAÇÃO ENV ====================

setup-env: ## Configura arquivo .env
	@echo "$(BLUE)⚙️  Configurando .env...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Criando .env a partir de .env.example...$(NC)"; \
		cp .env.example .env 2>/dev/null || echo "# CeciAI Environment Variables" > .env; \
		echo "" >> .env; \
		echo "# Gerado automaticamente em $$(date)" >> .env; \
		echo "ENV=development" >> .env; \
		echo "LOG_LEVEL=INFO" >> .env; \
		echo "" >> .env; \
		echo "# CoinAPI" >> .env; \
		echo "COINAPI_KEY=$${COINAPI_KEY:-your-api-key-here}" >> .env; \
		echo "COINAPI_MODE=development" >> .env; \
		echo "" >> .env; \
		echo "# Ollama" >> .env; \
		echo "OLLAMA_HOST=http://ollama:11434" >> .env; \
		echo "OLLAMA_MODEL_PRIMARY=llama3.2:3b" >> .env; \
		echo "OLLAMA_MODEL_CODE=llama3.2:1b" >> .env; \
		echo "OLLAMA_GPU_ENABLED=true" >> .env; \
		echo "OLLAMA_GPU_LAYERS=35" >> .env; \
		echo "" >> .env; \
		echo "# Trading" >> .env; \
		echo "INITIAL_CAPITAL=10000" >> .env; \
		echo "MAX_DAILY_LOSS=0.03" >> .env; \
		echo "MAX_POSITION_SIZE=0.20" >> .env; \
		echo "ENABLE_TRADING=false" >> .env; \
		echo "" >> .env; \
		echo "# Database" >> .env; \
		echo "POSTGRES_PASSWORD=changeme123" >> .env; \
		echo "REDIS_HOST=redis" >> .env; \
		echo "" >> .env; \
		echo "# Hardware" >> .env; \
		echo "CPU_CORES=8" >> .env; \
		echo "RAM_GB=32" >> .env; \
		echo "GPU_ENABLED=true" >> .env; \
		echo "GPU_VRAM_GB=8" >> .env; \
		echo "$(GREEN)✅ .env criado$(NC)"; \
	else \
		echo "$(GREEN)✅ .env já existe$(NC)"; \
	fi

# ==================== OLLAMA ====================

install-ollama: ## Instala e configura Ollama
	@echo "$(BLUE)📦 Instalando Ollama...$(NC)"
	@if command -v ollama >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Ollama já instalado$(NC)"; \
		ollama --version; \
	else \
		echo "$(YELLOW)Instalando Ollama...$(NC)"; \
		if [ "$$(uname)" = "Linux" ]; then \
			curl -fsSL https://ollama.ai/install.sh | sh; \
		elif [ "$$(uname)" = "Darwin" ]; then \
			brew install ollama 2>/dev/null || { \
				echo "$(RED)❌ Instale Homebrew primeiro: https://brew.sh$(NC)"; \
				exit 1; \
			}; \
		else \
			echo "$(YELLOW)⚠️  Windows detectado. Baixe manualmente: https://ollama.ai/download$(NC)"; \
			exit 0; \
		fi; \
		echo "$(GREEN)✅ Ollama instalado$(NC)"; \
	fi
	@$(MAKE) start-ollama
	@$(MAKE) pull-models

start-ollama: ## Inicia serviço Ollama
	@echo "$(BLUE)🚀 Iniciando Ollama...$(NC)"
	@if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Ollama já está rodando$(NC)"; \
	else \
		if [ "$$(uname)" = "Linux" ]; then \
			if command -v systemctl >/dev/null 2>&1; then \
				sudo systemctl start ollama 2>/dev/null || nohup ollama serve > /tmp/ollama.log 2>&1 & \
			else \
				nohup ollama serve > /tmp/ollama.log 2>&1 & \
			fi; \
		else \
			nohup ollama serve > /tmp/ollama.log 2>&1 & \
		fi; \
		sleep 3; \
		if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then \
			echo "$(GREEN)✅ Ollama iniciado$(NC)"; \
		else \
			echo "$(RED)❌ Erro ao iniciar Ollama$(NC)"; \
			exit 1; \
		fi; \
	fi

pull-models: ## Baixa modelos LLM otimizados
	@echo "$(BLUE)📥 Baixando modelos LLM...$(NC)"
	@echo "$(YELLOW)Modelo 1: llama3.2:3b (recomendado - 2GB)$(NC)"
	@ollama pull llama3.2:3b 2>/dev/null || echo "$(YELLOW)⚠️  Erro ao baixar llama3.2:3b$(NC)"
	@echo "$(YELLOW)Modelo 2: llama3.2:1b (ultra-rápido - 1GB)$(NC)"
	@ollama pull llama3.2:1b 2>/dev/null || echo "$(YELLOW)⚠️  Erro ao baixar llama3.2:1b$(NC)"
	@echo "$(GREEN)✅ Modelos baixados$(NC)"

validate-ollama: ## Valida instalação do Ollama
	@echo "$(BLUE)🧪 Validando Ollama...$(NC)"
	@if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then \
		echo "$(RED)❌ Ollama não está respondendo$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Ollama funcionando$(NC)"
	@echo "$(BLUE)Modelos instalados:$(NC)"
	@ollama list 2>/dev/null || echo "$(YELLOW)⚠️  Nenhum modelo instalado$(NC)"
	@echo "$(BLUE)Testando modelo...$(NC)"
	@RESPONSE=$$(ollama run llama3.2:3b "What is 2+2? Answer with just the number." 2>/dev/null | head -1); \
	if echo "$$RESPONSE" | grep -q "4"; then \
		echo "$(GREEN)✅ Modelo funcionando corretamente$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Resposta inesperada: $$RESPONSE$(NC)"; \
	fi

# ==================== DOCKER ====================

docker-build: ## Build da imagem Docker
	@echo "$(BLUE)🐳 Building Docker image...$(NC)"
	@$(DOCKER_COMPOSE) build --no-cache
	@echo "$(GREEN)✅ Imagem construída$(NC)"

docker-build-fast: ## Build rápido (com cache)
	@echo "$(BLUE)🐳 Building Docker image (cached)...$(NC)"
	@$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✅ Imagem construída$(NC)"

docker-up: ## Sobe todos os containers
	@echo "$(BLUE)🚀 Iniciando containers...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✅ Containers iniciados$(NC)"
	@sleep 5
	@$(DOCKER_COMPOSE) ps

docker-down: ## Para todos os containers
	@echo "$(YELLOW)🛑 Parando containers...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✅ Containers parados$(NC)"

docker-restart: ## Reinicia containers
	@$(MAKE) docker-down
	@$(MAKE) docker-up

docker-logs: ## Mostra logs
	@$(DOCKER_COMPOSE) logs -f

docker-logs-app: ## Logs da aplicação
	@$(DOCKER_COMPOSE) logs -f ceciai

db-up: ## Sobe o banco PostgreSQL e valida conexão
	@echo "$(BLUE)🐘 Subindo banco PostgreSQL...$(NC)"
	@$(DOCKER_COMPOSE) up -d postgres
	@echo "$(YELLOW)⏳ Aguardando banco inicializar...$(NC)"
	@sleep 10
	@echo "$(BLUE)🔍 Validando conexão com o banco...$(NC)"
	@for i in 1 2 3 4 5; do \
		if $(DOCKER_COMPOSE) exec -T postgres pg_isready -U ceciuser -d ceciai >/dev/null 2>&1; then \
			echo "$(GREEN)✅ PostgreSQL está rodando e aceitando conexões!$(NC)"; \
			echo "$(BLUE)📊 Status do banco:$(NC)"; \
			$(DOCKER_COMPOSE) exec -T postgres psql -U ceciuser -d ceciai -c "SELECT version();" 2>/dev/null | head -3 || true; \
			echo "$(BLUE)🔗 Conexão: postgresql://ceciuser:***@localhost:5432/ceciai$(NC)"; \
			exit 0; \
		else \
			echo "$(YELLOW)⏳ Tentativa $$i/5 - Aguardando banco...$(NC)"; \
			sleep 5; \
		fi; \
	done; \
	echo "$(RED)❌ Erro: PostgreSQL não respondeu após 5 tentativas$(NC)"; \
	echo "$(YELLOW)💡 Verifique os logs: make docker-logs postgres$(NC)"; \
	exit 1

db-down: ## Para o banco PostgreSQL
	@echo "$(YELLOW)🛑 Parando PostgreSQL...$(NC)"
	@$(DOCKER_COMPOSE) stop postgres
	@echo "$(GREEN)✅ PostgreSQL parado$(NC)"

db-logs: ## Mostra logs do PostgreSQL
	@$(DOCKER_COMPOSE) logs -f postgres

db-shell: ## Shell no PostgreSQL
	@$(DOCKER_COMPOSE) exec postgres psql -U ceciuser -d ceciai

db-ping: ## Testa conexão com o banco
	@echo "$(BLUE)🔍 Testando conexão com PostgreSQL...$(NC)"
	@if $(DOCKER_COMPOSE) exec -T postgres pg_isready -U ceciuser -d ceciai >/dev/null 2>&1; then \
		echo "$(GREEN)✅ PostgreSQL está respondendo$(NC)"; \
		$(DOCKER_COMPOSE) exec -T postgres psql -U ceciuser -d ceciai -c "SELECT 'Ping successful!' as status, now() as timestamp;" 2>/dev/null || true; \
	else \
		echo "$(RED)❌ PostgreSQL não está respondendo$(NC)"; \
		exit 1; \
	fi

# ==================== VERIFICAÇÃO ====================

verify-deployment: ## Verifica se deployment está OK
	@echo "$(BLUE)🔍 Verificando deployment...$(NC)"
	@sleep 5
	@echo "$(BLUE)Verificando containers...$(NC)"
	@$(DOCKER_COMPOSE) ps | grep -q "Up" && echo "$(GREEN)✅ Containers rodando$(NC)" || echo "$(RED)❌ Containers não estão rodando$(NC)"
	@echo "$(BLUE)Verificando API...$(NC)"
	@curl -s http://localhost:8000/health >/dev/null 2>&1 && echo "$(GREEN)✅ API respondendo$(NC)" || echo "$(YELLOW)⚠️  API ainda não está pronta$(NC)"
	@echo "$(BLUE)Verificando Ollama (container)...$(NC)"
	@curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo "$(GREEN)✅ Ollama respondendo$(NC)" || echo "$(YELLOW)⚠️  Ollama ainda não está pronto$(NC)"
	@echo "$(BLUE)Verificando Redis...$(NC)"
	@$(DOCKER_COMPOSE) exec -T redis redis-cli ping 2>/dev/null | grep -q "PONG" && echo "$(GREEN)✅ Redis respondendo$(NC)" || echo "$(YELLOW)⚠️  Redis ainda não está pronto$(NC)"

show-info: ## Mostra informações do deployment
	@echo "$(GREEN)╔══════════════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║                                                      ║$(NC)"
	@echo "$(GREEN)║     📊 Informações do Deployment                    ║$(NC)"
	@echo "$(GREEN)║                                                      ║$(NC)"
	@echo "$(GREEN)╚══════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(BLUE)🌐 URLs:$(NC)"
	@echo "  • API:           http://localhost:8000"
	@echo "  • Docs:          http://localhost:8000/docs"
	@echo "  • Health:        http://localhost:8000/health"
	@echo "  • Ollama:        http://localhost:11434"
	@echo "  • Dashboard:     http://localhost:8050 (se habilitado)"
	@echo ""
	@echo "$(BLUE)🐳 Containers:$(NC)"
	@$(DOCKER_COMPOSE) ps
	@echo ""
	@echo "$(BLUE)💡 Comandos úteis:$(NC)"
	@echo "  • Ver logs:      make logs"
	@echo "  • Parar:         make down"
	@echo "  • Reiniciar:     make restart"
	@echo "  • Shell:         make shell"
	@echo "  • Testes:        make test"
	@echo ""
	@echo "$(BLUE)🧪 Testar API:$(NC)"
	@echo "  curl -X POST http://localhost:8000/api/v1/analyze \\"
	@echo "    -H 'Content-Type: application/json' \\"
	@echo "    -d '{\"symbol\":\"BTC/USD\",\"timeframe\":\"1h\",\"strategy\":\"scalping\"}'"
	@echo ""

# ==================== DESENVOLVIMENTO ====================

install: ## Instala dependências localmente
	@echo "$(BLUE)📦 Instalando dependências...$(NC)"
	@pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependências instaladas$(NC)"

install-dev: ## Instala dependências de desenvolvimento
	@echo "$(BLUE)📦 Instalando dependências de dev...$(NC)"
	@pip install -r requirements.txt
	@pip install pytest pytest-cov pytest-asyncio pytest-mock ruff mypy ipython
	@echo "$(GREEN)✅ Dependências de dev instaladas$(NC)"

test: ## Roda testes
	@echo "$(BLUE)🧪 Rodando testes...$(NC)"
	@pytest tests/ -v --cov=. --cov-report=term-missing
	@echo "$(GREEN)✅ Testes concluídos$(NC)"

test-fast: ## Testes rápidos
	@pytest tests/ -v -x

coverage: ## Roda testes com cobertura completa
	@echo "$(BLUE)🧪 Rodando testes com cobertura...$(NC)"
	@pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html --cov-report=xml --cov-fail-under=100
	@echo "$(GREEN)✅ Cobertura de testes concluída$(NC)"
	@echo "$(BLUE)📊 Relatório HTML gerado em: htmlcov/index.html$(NC)"

lint: ## Roda linter e aplica correções automáticas
	@echo "$(BLUE)🔍 Executando lint com Ruff...$(NC)"
	@ruff check . --fix
	@echo "$(GREEN)✅ Lint concluído$(NC)"

format: ## Formata código com Ruff
	@echo "$(BLUE)✨ Formatando código com Ruff...$(NC)"
	@ruff format .
	@echo "$(GREEN)✅ Código formatado$(NC)"

format-lint: ## Formata código e executa lint (tudo com Ruff)
	@echo "$(BLUE)✨ Formatando e verificando código com Ruff...$(NC)"
	@echo "$(YELLOW)1/2 - Formatando código...$(NC)"
	@ruff format .
	@echo "$(YELLOW)2/2 - Executando lint e correções...$(NC)"
	@ruff check . --fix
	@echo "$(GREEN)✅ Formatação e lint concluídos$(NC)"

lint-check: ## Verifica lint sem fazer correções
	@echo "$(BLUE)🔍 Verificando lint (somente leitura)...$(NC)"
	@ruff check .
	@echo "$(GREEN)✅ Verificação de lint concluída$(NC)"

# ==================== SHELL ====================

shell: ## Shell no container
	@$(DOCKER_COMPOSE) exec ceciai /bin/bash

shell-ollama: ## Shell no Ollama
	@$(DOCKER_COMPOSE) exec ollama /bin/bash

shell-redis: ## Shell no Redis
	@$(DOCKER_COMPOSE) exec redis redis-cli

# ==================== LIMPEZA ====================

clean: ## Limpa containers e volumes
	@echo "$(YELLOW)🧹 Limpando...$(NC)"
	@$(DOCKER_COMPOSE) down -v
	@$(DOCKER) system prune -f
	@echo "$(GREEN)✅ Limpeza concluída$(NC)"

clean-all: ## Limpa tudo (incluindo imagens)
	@echo "$(RED)🧹 Limpando tudo...$(NC)"
	@$(DOCKER_COMPOSE) down -v --rmi all
	@$(DOCKER) system prune -af
	@rm -rf data/cache/* data/realtime/* logs/*.log
	@echo "$(GREEN)✅ Limpeza completa$(NC)"

clean-cache: ## Limpa cache Python
	@echo "$(YELLOW)🧹 Limpando cache...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✅ Cache limpo$(NC)"

# ==================== PRODUÇÃO ====================

prod-build: ## Build para produção
	@echo "$(BLUE)🏭 Building para produção...$(NC)"
	@$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml build --no-cache
	@echo "$(GREEN)✅ Build de produção concluído$(NC)"

prod-up: ## Sobe em produção
	@echo "$(BLUE)🚀 Iniciando em produção...$(NC)"
	@$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "$(GREEN)✅ Produção iniciada$(NC)"

prod-down: ## Para produção
	@$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml down

# ==================== BACKUP ====================

backup: ## Backup completo
	@echo "$(BLUE)💾 Criando backup...$(NC)"
	@mkdir -p backups
	@tar -czf backups/backup_$$(date +%Y%m%d_%H%M%S).tar.gz data/ logs/ .env
	@echo "$(GREEN)✅ Backup criado em backups/$(NC)"

# ==================== MONITORAMENTO ====================

stats: ## Estatísticas dos containers
	@$(DOCKER) stats

health: ## Health check
	@echo "$(BLUE)🏥 Health Check:$(NC)"
	@curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "$(RED)❌ API não está respondendo$(NC)"

logs: ## Logs de todos os containers
	@$(DOCKER_COMPOSE) logs -f

logs-tail: ## Últimas 100 linhas de logs
	@$(DOCKER_COMPOSE) logs --tail=100

# ==================== DADOS ====================

download-data: ## Baixa dados históricos
	@echo "$(BLUE)📥 Baixando dados históricos...$(NC)"
	@$(DOCKER_COMPOSE) exec ceciai python utils/download_historical_data.py
	@echo "$(GREEN)✅ Dados baixados$(NC)"

# ==================== ALIASES ====================

up: docker-up ## Alias para docker-up
down: docker-down ## Alias para docker-down
restart: docker-restart ## Alias para docker-restart
ps: ## Lista containers
	@$(DOCKER_COMPOSE) ps

# ==================== INFO ====================

version: ## Mostra versão
	@echo "CeciAI v0.3.0"
	@echo "Python: $$(python --version 2>&1)"
	@echo "Docker: $$(docker --version)"
	@echo "Docker Compose: $$(docker-compose --version)"

info: show-info ## Alias para show-info