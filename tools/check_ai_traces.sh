#!/bin/bash
#
# check_ai_traces.sh - AI痕迹分层检测脚本 v2.0
#
# 用法:
#   ./check_ai_traces.sh <markdown文件路径> [--json] [--lang cn|en|auto]
#
# 示例:
#   ./check_ai_traces.sh 2026_01/Article_CN.md
#   ./check_ai_traces.sh 2026_01/Article_EN.md --json --lang en
#
# 检测维度（分层）:
#   Layer 1: 基础统计 - Burstiness + 段落结构
#   Layer 2: 话术检测 - 中英文AI话术模式（15+种）
#   Layer 3: 词汇分析 - 高频词 + 同义词建议
#   Layer 4: 深度诊断 - 具体修复建议 + 位置定位
#
# 输出格式:
#   默认: 彩色终端输出
#   --json: JSON格式（供n8n工作流调用）
#
# 作者: Innora Security Research Team
# 版本: 2.0 | 日期: 2026-01-08
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 默认参数
JSON_OUTPUT=false
LANG_MODE="auto"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --lang)
            LANG_MODE="$2"
            shift 2
            ;;
        -h|--help)
            echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${BLUE}║           AI痕迹检测工具 v2.0                              ║${NC}"
            echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo -e "${GREEN}用法:${NC}"
            echo "  $0 <markdown文件路径> [--json] [--lang cn|en|auto]"
            echo ""
            echo -e "${GREEN}参数:${NC}"
            echo "  --json       输出JSON格式（供n8n工作流调用）"
            echo "  --lang       语言模式: cn（中文）, en（英文）, auto（自动检测）"
            echo ""
            echo -e "${GREEN}示例:${NC}"
            echo "  $0 2026_01/Article_CN.md"
            echo "  $0 2026_01/Article_EN.md --json --lang en"
            echo ""
            echo -e "${GREEN}检测维度（分层）:${NC}"
            echo "  Layer 1: 基础统计 - Burstiness + 段落结构"
            echo "  Layer 2: 话术检测 - 中英文AI话术模式（15+种）"
            echo "  Layer 3: 词汇分析 - 高频词 + 同义词建议"
            echo "  Layer 4: 深度诊断 - 具体修复建议 + 位置定位"
            exit 0
            ;;
        -*)
            echo -e "${RED}❌ 未知参数: $1${NC}"
            exit 1
            ;;
        *)
            FILE="$1"
            shift
            ;;
    esac
done

# 检查文件参数
if [ -z "$FILE" ]; then
    echo -e "${RED}❌ 错误: 请提供markdown文件路径${NC}"
    echo "用法: $0 <markdown文件路径> [--json] [--lang cn|en|auto]"
    exit 1
fi

# 检查文件是否存在
if [ ! -f "$FILE" ]; then
    echo -e "${RED}❌ 错误: 文件不存在: $FILE${NC}"
    exit 1
fi

# 自动检测语言
detect_language() {
    local file="$1"
    local cn_chars=$(grep -oE '[\x{4e00}-\x{9fff}]' "$file" 2>/dev/null | wc -l)
    local en_words=$(grep -oE '\b[a-zA-Z]+\b' "$file" 2>/dev/null | wc -l)

    if [ "$cn_chars" -gt "$en_words" ]; then
        echo "cn"
    else
        echo "en"
    fi
}

if [ "$LANG_MODE" = "auto" ]; then
    LANG_MODE=$(detect_language "$FILE")
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           AI痕迹检测报告                                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}文件: $FILE${NC}"
echo -e "${CYAN}时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo ""

TOTAL_SCORE=0
MAX_SCORE=100

# ═══════════════════════════════════════════════════════════════
# 1. Burstiness分析（句子长度变化）
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}1. Burstiness分析（句子长度变化）${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

BURST_RESULT=$(cat "$FILE" | awk 'BEGIN{FS=""}
/^[^#\|\-\*\`\>]/ && NF>5 {
    len=length($0);
    if(len>10) {
        count++; sum+=len; lens[count]=len
        if(len<min_len || min_len==0) min_len=len;
        if(len>max_len) max_len=len
    }
}
END {
    if(count>0) {
        avg=sum/count;
        for(i=1;i<=count;i++) variance+=(lens[i]-avg)^2
        stddev=sqrt(variance/count);
        burstiness=stddev/avg
        printf "%d|%.0f|%.0f|%.0f|%.0f|%.2f", count, avg, min_len, max_len, stddev, burstiness
    }
}')

IFS='|' read -r SENT_COUNT AVG_LEN MIN_LEN MAX_LEN STDDEV BURSTINESS <<< "$BURST_RESULT"

echo ""
echo "   句子总数:     $SENT_COUNT"
echo "   平均长度:     ${AVG_LEN}字符"
echo "   最短句:       ${MIN_LEN}字符"
echo "   最长句:       ${MAX_LEN}字符"
echo "   标准差:       $STDDEV"
echo ""
echo -n "   Burstiness指数: $BURSTINESS "

if (( $(echo "$BURSTINESS < 0.5" | bc -l) )); then
    echo -e "${RED}(偏低 - AI痕迹明显)${NC}"
    BURST_SCORE=15
elif (( $(echo "$BURSTINESS < 0.7" | bc -l) )); then
    echo -e "${YELLOW}(中等 - 有轻微AI痕迹)${NC}"
    BURST_SCORE=20
elif (( $(echo "$BURSTINESS < 0.9" | bc -l) )); then
    echo -e "${GREEN}(良好)${NC}"
    BURST_SCORE=25
else
    echo -e "${GREEN}(优秀 - 自然人类写作)${NC}"
    BURST_SCORE=30
fi

echo "   得分: ${BURST_SCORE}/30"
TOTAL_SCORE=$((TOTAL_SCORE + BURST_SCORE))

# ═══════════════════════════════════════════════════════════════
# 2. AI话术检测
# ═══════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}2. AI话术检测${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 检测各类AI话术（中英文分开）
declare -a PHRASE_DETAILS=()

if [ "$LANG_MODE" = "cn" ]; then
    # 中文AI话术检测（8种模式）
    PHRASE1=$(grep -cE "让我们深入|让我们来看|让我们探讨" "$FILE" 2>/dev/null) || PHRASE1=0
    PHRASE2=$(grep -cE "值得注意的是|值得一提" "$FILE" 2>/dev/null) || PHRASE2=0
    PHRASE3=$(grep -cE "综上所述|总的来说|总而言之" "$FILE" 2>/dev/null) || PHRASE3=0
    PHRASE4=$(grep -cE "不可否认|毫无疑问|显而易见" "$FILE" 2>/dev/null) || PHRASE4=0
    PHRASE5=$(grep -cE "首先.*其次|其次.*最后" "$FILE" 2>/dev/null) || PHRASE5=0
    PHRASE6=$(grep -cE "在当今.*时代|在.*的背景下" "$FILE" 2>/dev/null) || PHRASE6=0
    PHRASE7=$(grep -cE "我们可以得出|可以得出结论" "$FILE" 2>/dev/null) || PHRASE7=0
    PHRASE8=$(grep -cE "这意味着|这表明|这说明" "$FILE" 2>/dev/null) || PHRASE8=0

    echo "   检测到的AI话术（中文）:"
    echo "   - '让我们深入/探讨': $PHRASE1 处"
    echo "   - '值得注意的是': $PHRASE2 处"
    echo "   - '综上所述/总的来说': $PHRASE3 处"
    echo "   - '不可否认/毫无疑问': $PHRASE4 处"
    echo "   - '首先...其次...': $PHRASE5 处"
    echo "   - '在当今...时代': $PHRASE6 处"
    echo "   - '可以得出结论': $PHRASE7 处"
    echo "   - '这意味着/表明': $PHRASE8 处"

    AI_PHRASES=$((PHRASE1 + PHRASE2 + PHRASE3 + PHRASE4 + PHRASE5 + PHRASE6 + PHRASE7 + PHRASE8))

    # 记录位置用于Layer 4
    [ $PHRASE1 -gt 0 ] && PHRASE_DETAILS+=("让我们深入/探讨:$PHRASE1:建议改为直接陈述")
    [ $PHRASE2 -gt 0 ] && PHRASE_DETAILS+=("值得注意的是:$PHRASE2:建议删除或改为具体描述")
    [ $PHRASE3 -gt 0 ] && PHRASE_DETAILS+=("综上所述:$PHRASE3:建议改为'因此'或直接总结")
    [ $PHRASE4 -gt 0 ] && PHRASE_DETAILS+=("不可否认:$PHRASE4:建议删除虚词")
    [ $PHRASE5 -gt 0 ] && PHRASE_DETAILS+=("首先其次:$PHRASE5:建议使用1.2.3.或破折号列表")
    [ $PHRASE6 -gt 0 ] && PHRASE_DETAILS+=("在当今时代:$PHRASE6:建议删除或具体化时间")
    [ $PHRASE7 -gt 0 ] && PHRASE_DETAILS+=("可以得出结论:$PHRASE7:建议直接陈述结论")
    [ $PHRASE8 -gt 0 ] && PHRASE_DETAILS+=("这意味着:$PHRASE8:建议换用更具体动词")

else
    # 英文AI话术检测（8种模式）
    PHRASE1=$(grep -ciE "let's dive into|let's explore|let's take a look" "$FILE" 2>/dev/null) || PHRASE1=0
    PHRASE2=$(grep -ciE "it's worth noting|it is worth mentioning|notably" "$FILE" 2>/dev/null) || PHRASE2=0
    PHRASE3=$(grep -ciE "in conclusion|to summarize|to sum up|all in all" "$FILE" 2>/dev/null) || PHRASE3=0
    PHRASE4=$(grep -ciE "undeniably|undoubtedly|without a doubt|it goes without saying" "$FILE" 2>/dev/null) || PHRASE4=0
    PHRASE5=$(grep -ciE "firstly.*secondly|secondly.*finally|first.*then.*finally" "$FILE" 2>/dev/null) || PHRASE5=0
    PHRASE6=$(grep -ciE "in today's world|in the current landscape|in this day and age" "$FILE" 2>/dev/null) || PHRASE6=0
    PHRASE7=$(grep -ciE "we can conclude|it can be concluded|this suggests that" "$FILE" 2>/dev/null) || PHRASE7=0
    PHRASE8=$(grep -ciE "it's important to note|it is essential to|it should be noted" "$FILE" 2>/dev/null) || PHRASE8=0

    echo "   Detected AI phrases (English):"
    echo "   - 'Let's dive into/explore': $PHRASE1"
    echo "   - 'It's worth noting': $PHRASE2"
    echo "   - 'In conclusion/To sum up': $PHRASE3"
    echo "   - 'Undeniably/Undoubtedly': $PHRASE4"
    echo "   - 'Firstly...secondly...': $PHRASE5"
    echo "   - 'In today's world': $PHRASE6"
    echo "   - 'We can conclude': $PHRASE7"
    echo "   - 'It's important to note': $PHRASE8"

    AI_PHRASES=$((PHRASE1 + PHRASE2 + PHRASE3 + PHRASE4 + PHRASE5 + PHRASE6 + PHRASE7 + PHRASE8))

    # 记录位置用于Layer 4
    [ $PHRASE1 -gt 0 ] && PHRASE_DETAILS+=("Let's dive into:$PHRASE1:Replace with direct statement")
    [ $PHRASE2 -gt 0 ] && PHRASE_DETAILS+=("It's worth noting:$PHRASE2:Remove or state directly")
    [ $PHRASE3 -gt 0 ] && PHRASE_DETAILS+=("In conclusion:$PHRASE3:Replace with 'Therefore' or just conclude")
    [ $PHRASE4 -gt 0 ] && PHRASE_DETAILS+=("Undeniably:$PHRASE4:Remove filler words")
    [ $PHRASE5 -gt 0 ] && PHRASE_DETAILS+=("Firstly secondly:$PHRASE5:Use numbered or bulleted list")
    [ $PHRASE6 -gt 0 ] && PHRASE_DETAILS+=("In today's world:$PHRASE6:Remove or specify timeframe")
    [ $PHRASE7 -gt 0 ] && PHRASE_DETAILS+=("We can conclude:$PHRASE7:State conclusion directly")
    [ $PHRASE8 -gt 0 ] && PHRASE_DETAILS+=("It's important to note:$PHRASE8:Remove and state directly")
fi

echo ""
echo "   AI话术总计: $AI_PHRASES 处"

if [ $AI_PHRASES -eq 0 ]; then
    echo -e "   ${GREEN}✅ 优秀：无AI话术${NC}"
    PHRASE_SCORE=25
elif [ $AI_PHRASES -le 2 ]; then
    echo -e "   ${GREEN}良好：极少AI话术${NC}"
    PHRASE_SCORE=20
elif [ $AI_PHRASES -le 5 ]; then
    echo -e "   ${YELLOW}⚠️ 中等：存在一些AI话术${NC}"
    PHRASE_SCORE=15
else
    echo -e "   ${RED}❌ 警告：AI话术过多，建议修改${NC}"
    PHRASE_SCORE=10
fi

echo "   得分: ${PHRASE_SCORE}/25"
TOTAL_SCORE=$((TOTAL_SCORE + PHRASE_SCORE))

# ═══════════════════════════════════════════════════════════════
# 3. 高频词汇检测
# ═══════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}3. 高频词汇检测${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "   过度使用的词汇（出现5次以上）:"
OVERUSED=$(grep -oE "核心|关键|重要|显著|强大|全面|深入|完整|专业|高效|实现|提升" "$FILE" 2>/dev/null | sort | uniq -c | sort -rn | awk '$1>=5 {print "   - " $2 ": " $1 "次"}')

if [ -z "$OVERUSED" ]; then
    echo "   (无)"
    VOCAB_SCORE=25
else
    echo "$OVERUSED"
    OVERUSED_COUNT=$(echo "$OVERUSED" | wc -l)
    if [ $OVERUSED_COUNT -le 2 ]; then
        VOCAB_SCORE=20
    elif [ $OVERUSED_COUNT -le 4 ]; then
        VOCAB_SCORE=15
    else
        VOCAB_SCORE=10
    fi
fi

echo ""
echo "   得分: ${VOCAB_SCORE}/25"
TOTAL_SCORE=$((TOTAL_SCORE + VOCAB_SCORE))

# ═══════════════════════════════════════════════════════════════
# 4. 段落结构分析
# ═══════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}4. 段落结构分析${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

PARA_RESULT=$(cat "$FILE" | awk '
/^$/ { if(para_len>0) {
    total++
    if(para_len<100) short++
    else if(para_len<300) medium++
    else long++
    para_len=0
} next }
{ para_len += length($0) }
END {
    if(para_len>0) {
        total++
        if(para_len<100) short++
        else if(para_len<300) medium++
        else long++
    }
    printf "%d|%d|%d|%d", total, short, medium, long
}')

IFS='|' read -r TOTAL_PARA SHORT_PARA MEDIUM_PARA LONG_PARA <<< "$PARA_RESULT"

SHORT_PCT=$((SHORT_PARA * 100 / TOTAL_PARA))
MEDIUM_PCT=$((MEDIUM_PARA * 100 / TOTAL_PARA))
LONG_PCT=$((LONG_PARA * 100 / TOTAL_PARA))

echo "   段落总数: $TOTAL_PARA"
echo "   短段落(<100字符): $SHORT_PARA ($SHORT_PCT%)"
echo "   中段落(100-300字符): $MEDIUM_PARA ($MEDIUM_PCT%)"
echo "   长段落(>300字符): $LONG_PARA ($LONG_PCT%)"
echo ""

if [ $SHORT_PCT -gt 65 ]; then
    echo -e "   ${RED}❌ 警告：短段落过多，内容过于碎片化${NC}"
    STRUCT_SCORE=10
elif [ $SHORT_PCT -gt 50 ]; then
    echo -e "   ${YELLOW}⚠️ 中等：短段落偏多${NC}"
    STRUCT_SCORE=15
else
    echo -e "   ${GREEN}✅ 良好：段落结构均衡${NC}"
    STRUCT_SCORE=20
fi

echo "   得分: ${STRUCT_SCORE}/20"
TOTAL_SCORE=$((TOTAL_SCORE + STRUCT_SCORE))

# ═══════════════════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════════════════
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      检测结果汇总                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "   ┌───────────────────────┬──────────┐"
echo "   │ 维度                  │ 得分     │"
echo "   ├───────────────────────┼──────────┤"
printf "   │ %-21s │ %2d/30    │\n" "Burstiness(突发性)" "$BURST_SCORE"
printf "   │ %-21s │ %2d/25    │\n" "AI话术检测" "$PHRASE_SCORE"
printf "   │ %-21s │ %2d/25    │\n" "词汇多样性" "$VOCAB_SCORE"
printf "   │ %-21s │ %2d/20    │\n" "段落结构" "$STRUCT_SCORE"
echo "   ├───────────────────────┼──────────┤"
printf "   │ %-21s │ ${CYAN}%2d/100${NC}   │\n" "总分" "$TOTAL_SCORE"
echo "   └───────────────────────┴──────────┘"
echo ""

if [ $TOTAL_SCORE -ge 85 ]; then
    echo -e "   ${GREEN}██████████████████████ 优秀 (${TOTAL_SCORE}/100)${NC}"
    echo -e "   ${GREEN}文章自然度高，几乎无AI痕迹${NC}"
elif [ $TOTAL_SCORE -ge 70 ]; then
    echo -e "   ${GREEN}████████████████░░░░░░ 良好 (${TOTAL_SCORE}/100)${NC}"
    echo -e "   ${GREEN}文章整体自然，有少量可优化空间${NC}"
elif [ $TOTAL_SCORE -ge 55 ]; then
    echo -e "   ${YELLOW}██████████░░░░░░░░░░░░ 中等 (${TOTAL_SCORE}/100)${NC}"
    echo -e "   ${YELLOW}存在明显AI痕迹，建议优化${NC}"
else
    echo -e "   ${RED}██████░░░░░░░░░░░░░░░░ 需改进 (${TOTAL_SCORE}/100)${NC}"
    echo -e "   ${RED}AI痕迹严重，强烈建议重写${NC}"
fi

# ═══════════════════════════════════════════════════════════════
# Layer 4: 深度诊断 - 具体修复建议
# ═══════════════════════════════════════════════════════════════
if [ "$JSON_OUTPUT" = false ] && [ ${#PHRASE_DETAILS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}5. 深度诊断 - 具体修复建议${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    for detail in "${PHRASE_DETAILS[@]}"; do
        IFS=':' read -r phrase count suggestion <<< "$detail"
        echo -e "   ${RED}▸ $phrase${NC} (${count}处)"
        echo -e "     └─ ${GREEN}$suggestion${NC}"

        # 显示具体行号
        if [ "$LANG_MODE" = "cn" ]; then
            lines=$(grep -nE "$phrase" "$FILE" 2>/dev/null | head -3 | cut -d: -f1 | tr '\n' ',' | sed 's/,$//')
        else
            lines=$(grep -niE "$phrase" "$FILE" 2>/dev/null | head -3 | cut -d: -f1 | tr '\n' ',' | sed 's/,$//')
        fi
        if [ -n "$lines" ]; then
            echo -e "     └─ 行号: $lines"
        fi
        echo ""
    done
fi

echo ""
echo -e "${CYAN}提示: 运行 'cat HUMANIZED_WRITING_GUIDE.md' 查看完整优化指南${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════
# JSON输出格式（供n8n工作流调用）
# ═══════════════════════════════════════════════════════════════
if [ "$JSON_OUTPUT" = true ]; then
    # 确定评级
    if [ $TOTAL_SCORE -ge 85 ]; then
        GRADE="excellent"
        PASS="true"
    elif [ $TOTAL_SCORE -ge 70 ]; then
        GRADE="good"
        PASS="true"
    elif [ $TOTAL_SCORE -ge 55 ]; then
        GRADE="medium"
        PASS="false"
    else
        GRADE="poor"
        PASS="false"
    fi

    # 构建诊断详情JSON数组
    DIAGNOSTICS_JSON="["
    first=true
    for detail in "${PHRASE_DETAILS[@]}"; do
        IFS=':' read -r phrase count suggestion <<< "$detail"
        if [ "$first" = true ]; then
            first=false
        else
            DIAGNOSTICS_JSON+=","
        fi
        DIAGNOSTICS_JSON+="{\"phrase\":\"$phrase\",\"count\":$count,\"suggestion\":\"$suggestion\"}"
    done
    DIAGNOSTICS_JSON+="]"

    # 输出JSON
    cat <<EOF
{
  "file": "$FILE",
  "language": "$LANG_MODE",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "scores": {
    "burstiness": {
      "value": $BURSTINESS,
      "score": $BURST_SCORE,
      "max": 30
    },
    "ai_phrases": {
      "count": $AI_PHRASES,
      "score": $PHRASE_SCORE,
      "max": 25
    },
    "vocabulary": {
      "overused_count": ${OVERUSED_COUNT:-0},
      "score": $VOCAB_SCORE,
      "max": 25
    },
    "structure": {
      "short_pct": $SHORT_PCT,
      "medium_pct": $MEDIUM_PCT,
      "long_pct": $LONG_PCT,
      "score": $STRUCT_SCORE,
      "max": 20
    },
    "total": {
      "score": $TOTAL_SCORE,
      "max": 100
    }
  },
  "grade": "$GRADE",
  "pass": $PASS,
  "diagnostics": $DIAGNOSTICS_JSON,
  "statistics": {
    "sentence_count": $SENT_COUNT,
    "avg_sentence_length": $AVG_LEN,
    "paragraph_count": $TOTAL_PARA
  }
}
EOF
fi
