# 📡 Instrucciones para Acceso Remoto a la Landing Page

Hay varias formas de permitir que los alumnos accedan a la landing page desde otras redes:

## ✅ Opción 1: Usar ngrok (MÁS RECOMENDADA)

**Ngrok** es la herramienta más confiable para crear túneles públicos.

### Instalación:
```bash
# Descargar ngrok (Linux)
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# O descargar directamente
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin/
```

### Uso:
```bash
# 1. Registrarse en https://ngrok.com (gratis)
# 2. Obtener el token de autenticación
# 3. Configurar el token:
ngrok config add-authtoken TU_TOKEN_AQUI

# 4. Crear el túnel:
ngrok http 8000
```

**La URL pública aparecerá en la terminal** algo como:
```
Forwarding: https://abc123.ngrok.io -> http://localhost:8000
```

Los alumnos accederán a:
- **Landing simplificada:** `https://abc123.ngrok.io/landing-simplificada.html`
- **Landing original:** `https://abc123.ngrok.io/landing.html`

---

## ✅ Opción 2: Usar Serveo (GRATIS, SIN REGISTRO)

**Serveo** es súper simple y no requiere instalación ni registro.

```bash
# Simplemente ejecuta:
ssh -R 80:localhost:8000 serveo.net
```

La URL pública aparecerá inmediatamente en la terminal.

---

## ✅ Opción 3: Usar Cloudflare Tunnel (GRATIS, PROFESIONAL)

**Cloudflare Tunnel** es gratuito y muy estable.

### Instalación:
```bash
# Descargar cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

### Uso:
```bash
# Crear el túnel (sin necesidad de cuenta):
cloudflared tunnel --url http://localhost:8000
```

La URL pública aparecerá en la terminal.

---

## ✅ Opción 4: Acceso en Red Local (Sin Internet)

Si los alumnos están en la **misma red local** (misma WiFi):

### Tu IP local es: `21.0.0.140`

Los alumnos pueden acceder directamente a:
- `http://21.0.0.140:8000/landing-simplificada.html`
- `http://21.0.0.140:8000/landing.html`

**Nota:** Asegúrate de que el firewall permita conexiones al puerto 8000:
```bash
# En Ubuntu/Debian:
sudo ufw allow 8000/tcp

# En sistemas con firewalld:
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

---

## ✅ Opción 5: Subir a un Hosting Gratuito

Puedes subir los archivos HTML a servicios gratuitos:

### GitHub Pages:
```bash
# Ya están en tu repositorio, solo activa GitHub Pages:
# 1. Ve a Settings > Pages
# 2. Selecciona la rama y carpeta
# 3. La URL será: https://tuusuario.github.io/clase-experimentacion-cohorte-2025/landing-simplificada.html
```

### Netlify Drop:
1. Ve a https://app.netlify.com/drop
2. Arrastra y suelta `landing-simplificada.html`
3. Obtendrás una URL pública inmediatamente

### Vercel:
```bash
npm i -g vercel
vercel --prod
```

---

## 🎯 Recomendación

Para una demostración rápida en clase:
- **Mejor opción:** `ngrok` (más estable y profesional)
- **Opción rápida:** `serveo` (sin instalación)
- **Para producción:** GitHub Pages o Netlify

---

## 🔍 Verificar que el servidor HTTP está corriendo

```bash
# Ver si el servidor Python está activo:
ps aux | grep "python3 -m http.server"

# Si no está corriendo, iniciarlo:
cd /home/user/clase-experimentacion-cohorte-2025
python3 -m http.server 8000
```

---

## ❓ Troubleshooting

### El túnel no se conecta:
- Verifica tu conexión a internet
- Intenta otro servicio (ngrok, serveo, cloudflare)
- Verifica que el puerto 8000 esté libre: `netstat -tulpn | grep 8000`

### Los alumnos no pueden acceder:
- Verifica que el túnel esté activo
- Comparte la URL completa incluyendo el protocolo (https://)
- Asegúrate de incluir el nombre del archivo HTML al final

### Error de firewall en red local:
```bash
# Verificar firewall:
sudo ufw status

# Permitir puerto 8000:
sudo ufw allow 8000/tcp
```
