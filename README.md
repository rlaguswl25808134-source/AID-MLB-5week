# MLB 5주차 — 결정 트리 & 교차 검증/그리드 서치

결정 트리(05-1)와 교차 검증 · 그리드 서치(05-2)를 정리한 5주차 스터디 노트입니다.

- **사이트**: https://rlaguswl25808134-source.github.io/AID-MLB-5week/
- **노트**: [ML_Study_Week05.md](ML_Study_Week05.md)

## 구성

| 파일 | 설명 |
|---|---|
| `index.html` | GitHub Pages 배포용 단일 HTML (이미지 base64 내장) |
| `ML_Study_Week05.md` | Markdown 원본 노트 |
| `build_site_week5.py` | Markdown → HTML 생성 스크립트 |
| `assets/` | 지니 불순도 · 교차 검증 · 결정 트리 그림, 스터디 사진 |

## 이번 주 내용

- 결정 트리: 스무고개, 불순도(지니 · 엔트로피), 정보 이득
- 가지치기(`max_depth`)와 특성 중요도
- 검증 세트 · 교차 검증 · 그리드/랜덤 서치
- 실습: 캔 와인 분류 (알코올 · 당도 · pH → 레드/화이트)

## 다시 빌드하기

```powershell
python build_site_week5.py
```

`python build_site_week5.py`가 `index.html`을 생성합니다. GitHub Pages는 `main` 브랜치 루트를 사용합니다.
