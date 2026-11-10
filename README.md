# Elder Guard-AI

**Sistema inteligente de monitoramento e detecção de situações de emergência utilizando Visão Computacional e Inteligência Artificial.**

## Sobre o projeto

O **Elder Guard-AI** é um sistema de visão computacional desenvolvido para auxiliar no monitoramento de idosos que vivem sozinhos, com foco na identificação de situações potencialmente perigosas e na comunicação rápida com contatos de emergência.

O projeto utiliza técnicas de **Computer Vision, Pose Estimation e Machine Learning** para analisar movimentos e posturas corporais, buscando identificar eventos como quedas, possíveis desmaios e comportamentos que possam indicar situações de risco.

Ao detectar uma possível emergência, o sistema deverá gerar um alerta automático para um contato previamente cadastrado.

A proposta é desenvolver uma solução acessível, inteligente e não invasiva, priorizando a segurança, a privacidade e a autonomia dos usuários.

## Objetivos

- Desenvolver um sistema de monitoramento baseado em visão computacional.
- Identificar padrões de movimento associados a possíveis situações de emergência.
- Implementar algoritmos de detecção de quedas e análise de posturas corporais.
- Desenvolver um mecanismo automatizado de notificações.
- Reduzir falsos positivos por meio de análise temporal e validação de eventos.
- Explorar a aplicação prática de Inteligência Artificial em problemas relacionados à segurança e ao cuidado de idosos.

## Funcionalidades planejadas

| Funcionalidade | Descrição |
|---|---|
| Detecção de quedas | Identificação de movimentos bruscos e alterações de postura compatíveis com quedas. |
| Análise de imobilidade | Monitoramento de períodos prolongados sem movimentação após eventos suspeitos. |
| Identificação de possíveis desmaios | Reconhecimento de padrões corporais que possam indicar perda súbita de postura. |
| Reconhecimento de gestos | Identificação de movimentos potencialmente associados a pedidos de ajuda ou desconforto. |
| Monitoramento em tempo real | Processamento contínuo de imagens capturadas por uma câmera. |
| Notificações automáticas | Envio de alertas para contatos de emergência cadastrados. |
| Histórico de eventos | Registro de ocorrências identificadas durante o monitoramento. |

## Funcionamento

O sistema será desenvolvido com base em um pipeline de processamento de vídeo e análise de movimentos corporais.

**1. Captura de vídeo**

Uma câmera captura imagens do ambiente monitorado, fornecendo os dados necessários para o processamento.

**2. Detecção e estimativa de pose**

Modelos de visão computacional identificam a presença humana e estimam pontos-chave do corpo, permitindo acompanhar posições e movimentos.

**3. Análise temporal**

As informações extraídas de diferentes frames são analisadas para identificar mudanças bruscas de postura, movimentos incomuns e períodos de imobilidade.

**4. Classificação de eventos**

O sistema avalia os padrões identificados para determinar se existe uma possível situação de emergência.

**5. Geração de alertas**

Quando um evento suspeito atende aos critérios estabelecidos, uma notificação deverá ser enviada ao contato de emergência cadastrado.

## Tecnologias

A stack tecnológica será definida e atualizada conforme a evolução do projeto.

Tecnologias consideradas para o desenvolvimento:

| Tecnologia | Aplicação |
|---|---|
| Python | Desenvolvimento e integração dos componentes de IA. |
| OpenCV | Captura e processamento de vídeo. |
| MediaPipe / YOLO Pose | Detecção humana e estimativa de pose corporal. |
| PyTorch / Scikit-learn | Desenvolvimento e avaliação de modelos de classificação. |
| FastAPI | API para gerenciamento de eventos e notificações. |

*As tecnologias apresentadas estão em avaliação e podem ser alteradas durante o desenvolvimento.*

## Roadmap

### Fase 1 — Estruturação e protótipo

- [ ] Definir requisitos funcionais e não funcionais.
- [ ] Estruturar a arquitetura inicial.
- [ ] Implementar captura de vídeo.
- [ ] Integrar um modelo de estimativa de pose.
- [ ] Visualizar pontos-chave e movimentos corporais.

### Fase 2 — Detecção de eventos

- [ ] Desenvolver o algoritmo inicial de detecção de quedas.
- [ ] Implementar análise temporal de movimentos.
- [ ] Identificar períodos prolongados de imobilidade.
- [ ] Definir critérios de classificação de eventos suspeitos.
- [ ] Avaliar falsos positivos e falsos negativos.

### Fase 3 — Sistema de alertas

- [ ] Desenvolver uma API para gerenciamento de eventos.
- [ ] Implementar cadastro de contatos de emergência.
- [ ] Integrar serviço de notificações.
- [ ] Registrar histórico de ocorrências.

### Fase 4 — Validação e melhorias

- [ ] Construir cenários de testes controlados.
- [ ] Avaliar precisão, recall e tempo de resposta.
- [ ] Otimizar o processamento em tempo real.
- [ ] Implementar melhorias de privacidade e segurança.
- [ ] Documentar os resultados e limitações do sistema.

## Privacidade e segurança

O Elder Guard-AI considera a privacidade um requisito fundamental, especialmente por envolver monitoramento de ambientes domésticos.

O desenvolvimento deverá priorizar:

- Processamento local de imagens, sempre que viável.
- Minimização da coleta e do armazenamento de dados.
- Proteção das informações dos usuários e contatos de emergência.
- Controle de acesso aos recursos do sistema.
- Consentimento e tratamento adequado de dados pessoais, considerando a LGPD.

## Status

**Em desenvolvimento.**

O projeto encontra-se em fase inicial de planejamento, definição de arquitetura e avaliação das tecnologias que serão utilizadas.

## Limitações e responsabilidade

O Elder Guard-AI é um projeto experimental de pesquisa e desenvolvimento.

A identificação de quedas, imobilidade ou gestos suspeitos não constitui diagnóstico médico. O sistema poderá apresentar falsos positivos e falsos negativos, sendo necessária validação antes de qualquer utilização em ambientes reais.

A solução não substitui profissionais de saúde, equipamentos médicos certificados, supervisão humana ou serviços de emergência.

---

**Elder Guard-AI — Inteligência Artificial e Visão Computacional aplicadas à segurança e ao monitoramento assistido de idosos.**
