#!/bin/bash

# 配置
SERVICE_NAME="ccs"
API_URL="http://localhost:8501"
CHECK_INTERVAL=60  # 检查间隔（秒）
MAX_RETRIES=3      # 最大重试次数
LOG_FILE="/var/log/ccs_monitor.log"
ALERT_EMAIL=""     # 如果需要邮件通知，在这里填写邮箱地址

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查服务状态
check_service() {
    if ! systemctl is-active --quiet "$SERVICE_NAME"; then
        log "服务 $SERVICE_NAME 未运行"
        return 1
    fi
    return 0
}

# 检查API健康状况
check_api() {
    if ! curl -s --max-time 10 "$API_URL" > /dev/null; then
        log "API 健康检查失败: $API_URL"
        return 1
    fi
    return 0
}

# 重启服务
restart_service() {
    log "尝试重启服务 $SERVICE_NAME"
    systemctl restart "$SERVICE_NAME"
    sleep 30  # 等待服务完全启动
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log "服务重启成功"
        return 0
    else
        log "服务重启失败"
        return 1
    fi
}

# 发送告警
send_alert() {
    local message="$1"
    log "告警: $message"
    
    if [ -n "$ALERT_EMAIL" ]; then
        echo "$message" | mail -s "CCS Service Alert" "$ALERT_EMAIL"
    fi
}

# 主循环
main() {
    log "监控服务启动"
    
    while true; do
        service_ok=false
        api_ok=false
        retry_count=0
        
        # 检查服务和API状态
        if check_service; then
            service_ok=true
            if check_api; then
                api_ok=true
                log "服务运行正常"
            fi
        fi
        
        # 如果检查失败，进行重试
        while [ "$retry_count" -lt "$MAX_RETRIES" ] && { [ "$service_ok" = false ] || [ "$api_ok" = false ]; }; do
            log "检查失败，第 $((retry_count + 1)) 次重试"
            
            if restart_service; then
                if check_api; then
                    service_ok=true
                    api_ok=true
                    log "服务恢复正常"
                    break
                fi
            fi
            
            retry_count=$((retry_count + 1))
            [ "$retry_count" -lt "$MAX_RETRIES" ] && sleep 30
        done
        
        # 如果重试后仍然失败，发送告警
        if [ "$service_ok" = false ] || [ "$api_ok" = false ]; then
            send_alert "服务 $SERVICE_NAME 在 $MAX_RETRIES 次重试后仍然失败"
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

# 确保只运行一个实例
if pidof -o %PPID -x "$(basename "$0")" > /dev/null; then
    log "监控脚本已在运行"
    exit 1
fi

# 创建日志文件
touch "$LOG_FILE"
chmod 644 "$LOG_FILE"

# 启动主循环
main
