#!/bin/bash

echo "============================================"
echo "🌐 Compartir Landing Page con Alumnos"
echo "============================================"
echo ""
echo "Selecciona una opción:"
echo ""
echo "1) Usar Serveo (más rápido, sin instalación)"
echo "2) Usar Ngrok (más estable, requiere cuenta gratuita)"
echo "3) Usar Cloudflare Tunnel (profesional, sin cuenta)"
echo "4) Mostrar IP local (solo para misma red WiFi)"
echo "5) Salir"
echo ""
read -p "Opción: " option

case $option in
    1)
        echo ""
        echo "🚀 Iniciando túnel con Serveo..."
        echo ""
        echo "IMPORTANTE: La URL pública aparecerá a continuación."
        echo "Copia la URL y compártela con tus alumnos agregando:"
        echo "  /landing-simplificada.html  al final"
        echo ""
        echo "Presiona Ctrl+C para detener el túnel."
        echo ""
        sleep 2
        ssh -R 80:localhost:8000 serveo.net
        ;;
    2)
        echo ""
        if ! command -v ngrok &> /dev/null; then
            echo "❌ Ngrok no está instalado."
            echo ""
            echo "Para instalarlo:"
            echo "1. Visita: https://ngrok.com/download"
            echo "2. Descarga e instala ngrok"
            echo "3. Regístrate en https://dashboard.ngrok.com/signup"
            echo "4. Ejecuta: ngrok config add-authtoken TU_TOKEN"
            echo "5. Vuelve a ejecutar este script"
        else
            echo "🚀 Iniciando túnel con Ngrok..."
            echo ""
            ngrok http 8000
        fi
        ;;
    3)
        echo ""
        if ! command -v cloudflared &> /dev/null; then
            echo "❌ Cloudflare Tunnel no está instalado."
            echo ""
            echo "Para instalarlo:"
            echo "wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
            echo "sudo dpkg -i cloudflared-linux-amd64.deb"
            echo ""
            read -p "¿Quieres que lo instale automáticamente? (s/n): " install
            if [ "$install" = "s" ]; then
                wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
                sudo dpkg -i cloudflared-linux-amd64.deb
                rm cloudflared-linux-amd64.deb
                echo "✅ Instalado correctamente"
                sleep 2
                cloudflared tunnel --url http://localhost:8000
            fi
        else
            echo "🚀 Iniciando túnel con Cloudflare..."
            echo ""
            cloudflared tunnel --url http://localhost:8000
        fi
        ;;
    4)
        echo ""
        LOCAL_IP=$(hostname -I | awk '{print $1}')
        echo "📡 Tu IP local es: $LOCAL_IP"
        echo ""
        echo "Los alumnos en la MISMA RED WiFi pueden acceder a:"
        echo ""
        echo "  📱 Landing simplificada:"
        echo "     http://$LOCAL_IP:8000/landing-simplificada.html"
        echo ""
        echo "  📄 Landing original:"
        echo "     http://$LOCAL_IP:8000/landing.html"
        echo ""
        echo "⚠️  IMPORTANTE: Esto solo funciona si están conectados"
        echo "   a la misma red WiFi/LAN que tu computadora."
        echo ""
        echo "Si no funciona, verifica el firewall:"
        echo "  sudo ufw allow 8000/tcp"
        echo ""
        ;;
    5)
        echo "Saliendo..."
        exit 0
        ;;
    *)
        echo "Opción inválida"
        exit 1
        ;;
esac
