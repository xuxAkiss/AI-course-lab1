# -*- coding: utf-8 -*-
"""
实验一 数据清洗脚本
====================
用法:
    python preprocess.py            # 用默认配置清洗并生成 CSV
    import preprocess               # 实验代码里可直接复用 clean_text()

清洗步骤（均可开关，决策与理由见 实验决策记录.md）:
    1. STRIP_HEADERS   去除邮件头（默认开）
       KEEP_SUBJECT    是否保留 Subject 行（默认开）
    2. REMOVE_QUOTES   去除正文中的引用行（默认关，留作实验变量）
    3. KEEP_ORIGINAL   同时输出原始文本列供对比（默认关）

输出:
    train_cleaned.csv / test_cleaned.csv
"""
import os
import re

import pandas as pd

# ---------------- 可配置项 ----------------
# 注: 当前默认值 = v1 配置 (实测最优, 见 实验决策记录.md 决策11)。
#     v2 实验(剥Subject标签/去Re:/删引用归属行)曾试过, 验证集 -0.66pp, 已回退。
STRIP_HEADERS = True     # 是否去除邮件头
KEEP_SUBJECT = True      # 是否保留 Subject 行
KEEP_SUBJECT_LABEL = True  # True=v1: 保留完整 "Subject: ..." 行(含标签和 Re:)
                           # False=v2: 剥标签去 Re: 链, 只留主题内容
CLEAN_ATTRIBUTION = False  # v2 实验项: 删除 "X writes:/wrote:" 引用归属行(实测有害)
REMOVE_QUOTES = False    # 是否去除正文中 ">" 开头的引用行
KEEP_ORIGINAL = False    # 输出文件里是否额外保留原始 text 列
# ------------------------------------------

# 邮件头字段: 单词/连字符/点 + 冒号。只从文本开头连续匹配，遇到正文行即停止。
HEADER_PATTERN = re.compile(r'^[A-Za-z][A-Za-z0-9.\-]*:\s')
SUBJECT_PATTERN = re.compile(r'^Subject:\s*(.*)$', re.IGNORECASE)
RE_PREFIX_PATTERN = re.compile(r'^(?:\s*Re:\s*)+', re.IGNORECASE)  # "Re: Re: X" -> "X"
# 纯引用归属行: 整行以 "writes:" / "wrote:" 结尾的废话行 ("In article <...> wrote:" 等)
ATTRIBUTION_PATTERN = re.compile(r'^.*\s(?:wrote|writes):\s*$', re.IGNORECASE)
QUOTE_PATTERN = re.compile(r'^\s*(?:>|In article <|On \w+.*wrote:$)')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def clean_text(text: str) -> str:
    """按配置清洗单条文本。"""
    if not isinstance(text, str):
        return text

    lines = text.split('\n')
    if STRIP_HEADERS:
        lines = _strip_headers(lines)
    if CLEAN_ATTRIBUTION:
        lines = [ln for ln in lines if not ATTRIBUTION_PATTERN.match(ln.strip())]

    body = '\n'.join(lines).strip()
    if REMOVE_QUOTES:
        kept = [ln for ln in body.split('\n') if not QUOTE_PATTERN.match(ln)]
        body = '\n'.join(kept).strip()
    return body


def _strip_headers(lines):
    """从头开始剥离连续的头字段行（含续行）。

    保留 Subject 时: 剥掉 "Subject:" 标签和 "Re:" 前缀链, 只留主题内容。
    """
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if HEADER_PATTERN.match(line):
            if KEEP_SUBJECT:
                sm = SUBJECT_PATTERN.match(line)
                if sm:
                    if KEEP_SUBJECT_LABEL:   # v1: 整行保留(含 "Subject:" 标签和 "Re:")
                        out.append(line)
                    else:                     # v2: 剥标签 + 去 Re: 链, 只留主题内容
                        content = RE_PREFIX_PATTERN.sub('', sm.group(1)).strip()
                        if content:
                            out.append(content)
            i += 1
            # 跳过续行（以空格/Tab 开头）
            while i < len(lines) and lines[i][:1] in (' ', '\t'):
                i += 1
        else:
            break
    return out + lines[i:]


def load_raw():
    """读取原始 CSV（与 draft_main.py 的读取方式保持一致）。"""
    train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'test_data_unlabeled.csv'))
    return train_df, test_df


def main():
    train_df, test_df = pd.read_csv(os.path.join(BASE_DIR, 'train_data.csv')), \
        pd.read_csv(os.path.join(BASE_DIR, 'test_data_unlabeled.csv'))

    print('清洗前:')
    print(f'  train {train_df.shape[0]} 条, 平均 {train_df["text"].astype(str).str.len().mean():.0f} 字符')
    print(f'  test  {test_df.shape[0]} 条, 平均 {test_df["text"].astype(str).str.len().mean():.0f} 字符')

    # 抽样统计头部剥离命中率
    sample = train_df['text'].astype(str).head(500)
    hit = sample.map(lambda s: HEADER_PATTERN.match(s.strip()) is not None).mean()
    print(f'  (前500条中以邮件头开头的比例: {hit:.1%})')

    for name, df, cols in [('train', train_df, ['text', 'target']),
                           ('test', test_df, ['text'])]:
        cleaned = df.copy()
        cleaned['text'] = df['text'].astype(str).map(clean_text)
        if KEEP_ORIGINAL:
            cleaned.insert(1 if 'target' in cleaned else 1, 'text_raw', df['text'])
        cleaned.to_csv(os.path.join(BASE_DIR, f'{name}_cleaned.csv'), index=False)
        print(f'  -> 已保存 {name}_cleaned.csv, 平均 {cleaned["text"].str.len().mean():.0f} 字符, 行数 {len(cleaned)}')

    print('\n清洗后示例(前150字符):')
    print('  ' + repr(clean_text(train_df['text'].astype(str).iloc[0])[:150]))
    print('  ' + repr(clean_text(train_df['text'].astype(str).iloc[2])[:150]))
    print('  ' + repr(clean_text(test_df['text'].astype(str).iloc[0])[:150]))


if __name__ == '__main__':
    main()
