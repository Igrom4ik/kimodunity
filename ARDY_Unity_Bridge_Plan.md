# ARDY ⇄ Unity 6: интерактивная генерация анимаций, live-предпросмотр и бэйк

Версия плана: 1.0 · дата: 2026-08-08
Источники фактов: `C:\Nvidia_Ardy\ardy` (NVIDIA ARDY, Apache-2.0), `E:\Unity\ArdyUnity` (Unity 6000.5.3f1)

---

## 0. Проверенные факты (не предположения)

Всё ниже прочитано в коде/ассетах, ссылки — на реальные файлы.

### 0.1 Что отдаёт модель

`ArdyMotionRep.inverse()` (`ardy/motion_rep/reps/ardy_motionrep.py:238`) возвращает dict:

| Ключ | Форма | Смысл |
|---|---|---|
| `local_rot_mats` | `[B,T,J,3,3]` | локальные повороты (относительно родителя) |
| `global_rot_mats` | `[B,T,J,3,3]` | глобальные повороты |
| `posed_joints` | `[B,T,J,3]` | мировые позиции суставов, метры |
| `root_positions` | `[B,T,3]` | мировая позиция таза, метры |
| `foot_contacts` | `[B,T,4]` bool | `[L_heel, L_toe, R_heel, R_toe]` |
| `global_root_heading` | `[B,T,2]` | `[cos θ, sin θ]` |

Ключевое: **есть и локальные, и глобальные повороты, и контакты стоп**. Контакты — бесплатный вход для IK-фиксации ног при бэйке; в черновике этот канал не использовался вообще.

### 0.2 Система координат ARDY — доказательство

Три независимых подтверждения, что ARDY = **правая (RH) система, Y-вверх, forward = +Z, left = +X, метры, пол на y=0**:

1. **Ассет скелета.** Дамп `cskel27/joints.p`: `LeftArm` в `+X`, `RightArm` в `−X`, ноги вниз по `−Y`, носки вперёд в `+Z`.
2. **Конвертер MuJoCo** (`scripts/interactive_demo/motion_io.py:169-171`): `mujoco_to_ardy = Rx(−90°) · Rz(−90°)`. MuJoCo Z-up/X-forward/Y-left (RH) → X_mj(forward) в +Z, Z_mj(up) в +Y, Y_mj(left) в +X.
3. **Heading** (`ardy/motion_rep/tools.py:126`): `atan2(diff_z, −diff_x)` от вектора `r_hip − l_hip`; θ=0 ⇔ right_hip в −X. Матрица `corrective_mat_Y` (`tools.py:169`) — каноническая RH-матрица поворота вокруг +Y. `scripts/generate.py:246`: `first_heading_angle = zeros # facing +Z`.

### 0.3 Rest-поза cskel27 — это ТОЧНАЯ T-поза

Дамп `neutral_joints` (метры, мировые координаты, Hips в начале координат):

```
 0 Hips            ( 0.0000,  0.0000,  0.0000)
 1 Spine           ( 0.0000,  0.0710, -0.0473)
 2 Spine1          ( 0.0000,  0.1642, -0.0638)
 3 Spine2          ( 0.0000,  0.2585, -0.0720)
 4 Spine3          ( 0.0000,  0.3531, -0.0720)
 5 Neck            ( 0.0000,  0.6016, -0.0365)
 6 Head            ( 0.0000,  0.7298, -0.0139)
 7 RightShoulder   (-0.0320,  0.5259, -0.0187)
 8 RightArm        (-0.1909,  0.5259, -0.0187)   off = (-0.1589, 0, 0)
 9 RightForeArm    (-0.4863,  0.5259, -0.0187)   off = (-0.2954, 0, 0)
10 RightHand       (-0.7190,  0.5259, -0.0187)   off = (-0.2327, 0, 0)
11 RightHandEnd    (-0.7886,  0.5259, -0.0187)
12 RightHandThumb1 (-0.7468,  0.5074,  0.0277)
13 LeftShoulder    ( 0.0320,  0.5259, -0.0187)
14 LeftArm         ( 0.1909,  0.5259, -0.0187)
15 LeftForeArm     ( 0.4863,  0.5259, -0.0187)
16 LeftHand        ( 0.7190,  0.5259, -0.0187)
17 LeftHandEnd     ( 0.7886,  0.5259, -0.0187)
18 LeftHandThumb1  ( 0.7468,  0.5074,  0.0277)
19 RightUpLeg      (-0.0949, -0.0277,  0.0000)
20 RightLeg        (-0.0949, -0.4398,  0.0000)
21 RightFoot       (-0.0949, -0.8959,  0.0000)
22 RightToeBase    (-0.0949, -0.9544,  0.1607)
23 LeftUpLeg       ( 0.0949, -0.0277,  0.0000)
24 LeftLeg         ( 0.0949, -0.4398,  0.0000)
25 LeftFoot        ( 0.0949, -0.8959,  0.0000)
26 LeftToeBase     ( 0.0949, -0.9544,  0.1607)
```

Три следствия, на которых держится вся архитектура:

* **Единичные локальные повороты = идеальная T-поза.** Значит `AvatarBuilder.BuildHumanAvatar` примет этот риг без ручной настройки — Unity требует именно T-позу.
* **В rest-позе фрейм каждого сустава совпадает с мировым** (все локальные повороты единичны ⇒ все глобальные тоже). Поэтому одно и то же преобразование координат применяется к *любому* повороту — и локальному, и глобальному — без per-bone поправок. Это радикально упрощает конвертацию (см. §2.3).
* **Rest-поза донора — сама референс-поза модели**, значит `global_rot_mats` из ARDY — это уже готовая мировая дельта от rest (`R_s0 = I`), что делает ретаргет-математику тривиальной.

Рост скелета ≈ 1.75 м (макушка ~0.80 при стопах на −0.95). Метры совпадают с Unity 1:1, масштабирование не нужно.

### 0.4 Модели и бюджет латентности

`ardy/model/registry.py:24`:

| Модель | Скелет | FPS | Horizon (кадров) | Длительность чанка |
|---|---|---|---|---|
| `ARDY-Core-RP-20FPS-Horizon40` | cskel27 | 20 | 40 | **2.00 с** |
| `ARDY-Core-RP-20FPS-Horizon8` | cskel27 | 20 | 8 | **0.40 с** |
| `ARDY-G1-RP-25FPS-Horizon52` | g1skel34 | 25 | 52 | 2.08 с |
| `ARDY-G1-RP-25FPS-Horizon8` | g1skel34 | 25 | 8 | 0.32 с |

SOMA-модели заявлены как «coming soon» — в реестре их нет, `somaskel30/77` присутствуют только как определения скелета для парсинга BVH-датасета.

**Вывод: рабочая связка — `core8` для интерактива, `core40` для финального качества.** G1 — робот Unitree, не человек: суставы pitch/roll/yaw-цепочками, `RightHand` там отсутствует как понятие. Для гуманоидных персонажей Unity он непригоден.

Реальное время достижимо тогда и только тогда, когда время одного `autoregressive_step` < длительности чанка (0.40 с для core8). Это **измеряемая величина**, а не допущение — демо уже печатает `Generate step time` (`generation.py:415`). Спайк S0 (§8) — первое, что нужно сделать.

### 0.5 Что уже готово в ARDY и переиспользуется

| Возможность | Где | Комментарий |
|---|---|---|
| Автогрессивный шаг | `Ardy.autoregressive_step` (`ardy/model/ardy_model.py:710`) | принимает `init_history_sequence`, `text_feat`, `motion_mask`/`observed_motion` |
| Однократная генерация | `Ardy.__call__` (`ardy_model.py:535`) | используется в `scripts/generate.py` |
| Загрузка модели | `load_model()` (`ardy/model/load_model.py:159`) | автоскачивание с HF или `CHECKPOINTS_DIR` |
| Постобработка (foot skating) | `post_process_motion()` (`ardy/postprocess.py:184`) | принимает `contacts`, возвращает исправленные `local_rot_mats`/`root_positions` |
| Экспорт BVH | `save_bvh_file()` (`ardy/skeleton/bvh.py:582`) | `ZXY`-эйлеры, по умолчанию **масштабирует в сантиметры** |
| Экспорт BVH из демо | `export_bvh()` (`scripts/interactive_demo/session_io.py:184`) | готовая кнопка в UI |
| Окно истории | `compute_window_num_frames()` (`scripts/interactive_demo/window_budget.py`) | модель обучена на окне ≤ 10 с, без кропа истории деградирует в джиттер |
| Кэш текстовых эмбеддингов | `CachedTextEncoder` (`interactive_demo/embedding_cache.py`) | дисковый кэш, критичен для интерактива |

### 0.6 Ограничение, которое ломает «headless» сценарий

Демо создаёт сессию **только на подключение viser-клиента** (`client.py:24` `on_client_connect` → `client_sessions[client.client_id] = session`) и удаляет её на disconnect (`client.py:448`). Без открытой вкладки браузера генерации не будет. Плюс `--offload` вызывает `memory_manager.purge_encoder_completely()` после каждого шага (`generation.py:419`) — при интерактивной смене промптов это лишние перезагрузки энкодера.

Отсюда двухфазная стратегия сервиса (§1.2).

---

## 1. Архитектура

### 1.1 Два контура, а не один

Черновик смешивал стриминг и бэйк в один поток. Правильно разделить:

```
                    ┌──────────────────────── PYTHON (AI) ────────────────────────┐
                    │  ARDY: load_model → autoregressive_step → motion_rep.inverse│
                    │        → post_process_motion → BridgeServer (TCP)           │
                    └──────────────┬──────────────────────────────┬───────────────┘
                                   │ hello (скелет)               │ chunk (кадры)
                                   ▼                              ▼
  ┌──────────────────────────────── C# / UNITY 6 ────────────────────────────────┐
  │                                                                              │
  │  КОНТУР A: LIVE                     КОНТУР B: BAKE                           │
  │  ArdyClient (TCP, фон. поток)  ──▶  MotionBuffer (полный ring всех кадров)   │
  │        │ lock-free очередь                    │                              │
  │        ▼                                      ▼                              │
  │  DonorRig (генерируется из hello)      HumanPoseHandler-семплинг              │
  │        │ localRotation = f(quat)              │                              │
  │        ▼                                      ▼                              │
  │  Avatar (AvatarBuilder)                 AnimationClip (muscle-кривые)        │
  │        │ HumanPoseHandler                     │                              │
  │        ▼                                      ▼                              │
  │  Целевой персонаж (любой Humanoid)      .anim в AssetDatabase                │
  └──────────────────────────────────────────────────────────────────────────────┘
```

Контур A ест данные «как приходят» (может пропускать кадры, интерполировать, отставать). Контур B — единственный источник истины для бэйка, ест **все** кадры без потерь, из локального буфера, вне зависимости от того, что показывает вьюпорт.

### 1.2 Фазы сервиса на стороне Python

* **Фаза 1 — `BridgeMixin` в существующее демо.** Класс подмешивается в `InteractiveTimelineDemo` (`scripts/run_demo.py:52`), хук после `_generate_step` и в `set_frame`. Плюсы: сразу получаем промпты по таймлайну, waypoints, kinematic constraints, постобработку, GUI. Минус: зависимость от внутренностей демо + требуется открытая вкладка браузера.
* **Фаза 2 — `scripts/run_bridge.py`, headless.** Собственный автогрессивный цикл (~200 строк по образцу `_generate_step`), зависящий только от `ardy.model.load_model`, `ardy.motion_rep`, `ardy.postprocess`. Ни viser, ни gradio, ни Node.js. Unity становится единственным UI.

Чтобы фаза 2 не была переписыванием: **весь код, не специфичный для демо, сразу кладём в отдельный пакет `ardy_bridge/`** (протокол, сериализация, TCP-сервер, преобразование выхода модели в wire-формат). Миксин фазы 1 — тонкая обёртка над ним.

### 1.3 Разделение ответственности (по вашему требованию: AI — Python, остальное — C#)

| Слой | Язык | Обоснование |
|---|---|---|
| Инференс, диффузия, постобработка | Python | Это и есть AI-часть |
| Матрица→кватернион | Python | Сериализация, не «конвертация координат»; `scipy.Rotation.as_quat()` уже в зависимостях, векторизован, численно устойчив |
| **Конвертация координат RH→LH** | **C#** | Ваше требование; проверяется golden-векторами из Python |
| Скелет, аватар, ретаргет, бэйк, UI | C# | Полностью |

---

## 2. Математика конвертации ARDY → Unity

### 2.1 Выбор отражения — это НЕ `−Z`

Черновик предлагал `(x, y, −z)` и кватернион `(−x, −y, z, w)`. Математически это валидная RH→LH-конвертация, но она разворачивает персонажа на 180° относительно Unity-forward: ARDY-forward `+Z` уедет в Unity `−Z`. Все waypoints, направления взгляда и «идти вперёд» окажутся задом наперёд относительно мировых осей Unity.

Правильное отражение — **по X**:

| ARDY | | Unity |
|---|---|---|
| left = `+X` | → | `−X` = left персонажа в Unity ✔ |
| up = `+Y` | → | `+Y` ✔ |
| forward = `+Z` | → | `+Z` ✔ |

```csharp
// S = diag(-1, 1, 1)
public static Vector3 ArdyToUnity(Vector3 p) => new Vector3(-p.x, p.y, p.z);

// Сопряжение R' = S·R·S для кватерниона (x,y,z,w):
public static Quaternion ArdyToUnity(Quaternion q) => new Quaternion(q.x, -q.y, -q.z, q.w);
```

**Проверка формулы.** ARDY: поворот +90° вокруг +Y (RH) переводит forward `+Z` → `+X` (это ARDY-left, т.е. персонаж повернулся налево). Кватернион `(0, sin45°, 0, cos45°)`. После формулы: `(0, −sin45°, 0, cos45°)` = Unity-поворот −90° вокруг Y, который переводит `+Z` → `−X` = Unity-left. Персонаж повернулся налево. ✔ Лево/право не зеркалятся.

Инверсия (Unity → ARDY, для отправки waypoints обратно) — та же формула, отражение инволютивно.

### 2.2 Почему хватает одной формулы на все кости

Обычно RH→LH-ретаргет требует per-bone поправок, потому что rest-фрейм каждой кости произволен. Здесь — нет: как показано в §0.3, в rest-позе ARDY **все локальные повороты единичны**, значит фрейм каждого сустава совпадает с мировым. Сопряжение `S·R·S` применяется единообразно к:

* локальным поворотам (фрейм родителя тоже отражён этим же `S`),
* глобальным поворотам,
* позициям суставов и корня,
* оффсетам костей rest-скелета.

Это надо зафиксировать в комментарии к коду — иначе через полгода кто-то «починит» это добавлением per-bone поправок.

### 2.3 Что именно ставим в трансформы

```
donorRoot                       (GameObject в начале координат сцены)
└── Hips        localPosition = ArdyToUnity(root_positions[t])
                localRotation = ArdyToUnity(quat(local_rot_mats[t,0]))
    └── Spine   localPosition = ArdyToUnity(neutral[1] - neutral[0])   ← константа, ставится один раз
                localRotation = ArdyToUnity(quat(local_rot_mats[t,1]))
    └── ...
```

`localPosition` дочерних костей — константы из `neutral_joints`, ставятся при генерации рига и больше не трогаются. Каждый кадр меняются только 27 `localRotation` + `Hips.localPosition`. Это ~28 записей в трансформы на кадр — ничтожно.

### 2.4 Единицы и земля

* ARDY: метры, пол на `y = 0`, таз в стойке на `y ≈ 0.90…0.95`.
* Unity: метры, пол на `y = 0`. **Масштаб 1:1, конвертация не нужна.**
* Внимание: `save_bvh_file(..., scale_to_cm=True)` — дефолт. Если пойдёте по BVH-пути, ставьте `scale_to_cm=False` либо `Scale Factor = 0.01` в импортере.

### 2.5 Golden-vector тесты (обязательно, не «если будет время»)

Python-скрипт `tools/gen_golden.py` формирует `golden_transform.json`:
* 8 опорных кватернионов (identity, ±90° вокруг каждой оси, случайный) в ARDY-пространстве и их Unity-эквиваленты;
* 1 кадр реальной генерации: `local_rot_mats`, `root_positions`, и — контрольно — `posed_joints`.

C#-тест (`ArdyUnity.Tests.EditMode`): применяет `ArdyToUnity`, строит донор-риг, прогоняет FK средствами Unity и сравнивает мировые позиции суставов с `ArdyToUnity(posed_joints)` с допуском 1e-4 м. Это ловит **все** ошибки знака, порядка и handedness одним тестом.

---

## 3. Протокол моста

### 3.1 Транспорт: TCP, а не UDP

Черновик предлагал UDP «из-за отсутствия задержек на подтверждение приёма». На loopback это ложная экономия:

* Поток автогрессивный: потеря чанка = дырка в анимации и рассинхрон буфера бэйка. UDP не даёт ни доставки, ни порядка.
* Чанк core40 = 40 кадров × (27 кватернионов + корень) × 4 байта ≈ **18 КБ**. Это 13 IP-фрагментов; потеря любого убивает всю датаграмму.
* На `127.0.0.1` TCP не платит RTT-штрафа — данные идут через loopback-буфер ядра. С `TCP_NODELAY` задержка на локалхосте — десятки микросекунд.

**Решение: TCP на `127.0.0.1:8801`, `NoDelay = true`, длино-префиксный фрейминг.**

### 3.2 Формат кадра

```
[4B] magic     = 0x59445241 ("ARDY", little-endian)
[2B] version   = 1
[2B] msgType   = uint16
[4B] jsonLen   = uint32
[4B] blobLen   = uint32
[N ] jsonHeader (UTF-8, компактный JSON)
[M ] blob       (float32 little-endian, плотный)
```

JSON для метаданных (читаемо, отлаживаемо, расширяемо), сырой `float32`-блоб для массивов (быстро, без парсинга). Читается в C# через `BinaryReader` + `MemoryMarshal.Cast<byte, float>`.

### 3.3 Сообщения Python → Unity

| msgType | Имя | Payload |
|---|---|---|
| 1 | `hello` | JSON: `{protocol, model, skeleton, fps, jointNames[], parents[], rootIdx, footContactJoints[], genHorizon, numFramesPerToken}`; blob: `neutral_joints` `[J,3]` float32 |
| 2 | `chunk` | JSON: `{startFrame, count, revision, hasContacts}`; blob: на кадр — `root[3] + quat[J*4] (+ contacts[4])` |
| 3 | `playhead` | JSON: `{frame, playing, maxFrame}` |
| 4 | `invalidate` | JSON: `{fromFrame}` — кадры от `fromFrame` устарели (произошёл replan/restart) |
| 5 | `status` | JSON: `{state, prompt, lastStepMs, vramMb, message}` |
| 6 | `error` | JSON: `{code, message}` |

`hello` несёт **полное описание скелета**, включая rest-оффсеты. Значит **Unity строит донорский риг автоматически**, ничего импортировать руками не нужно. Это снимает целый блок черновика («импорт пустого FBX-скелета, ручная настройка Humanoid, ручной словарь костей»).

`invalidate` обязателен: ARDY переписывает уже сгенерированные кадры при replan (`generation.py:372` — конкатенация `motion_tensor[:, :history_end_idx+1]` с новым окном). Без этого сообщения буфер бэйка накопит устаревшие кадры. **В черновике этого механизма нет — это тихий баг, который проявился бы только на длинных записях.**

### 3.4 Сообщения Unity → Python

| msgType | Имя | Payload |
|---|---|---|
| 100 | `setPrompt` | `{text, fromFrame}` |
| 101 | `transport` | `{action: play\|pause\|seek\|restart\|restartFromNow, frame}` |
| 102 | `setParams` | `{diffusionSteps, cfgText, cfgConstraint, numSamples, postprocess}` |
| 103 | `waypoint` | `{frame, x, z}` (в Unity-пространстве; Python конвертирует обратно) |
| 104 | `requestRange` | `{fromFrame, toFrame}` — дослать кадры после реконнекта |
| 105 | `bye` | — |

### 3.5 Размеры и пропускная способность

cskel27, float32: `3 + 27×4 + 4 = 115` float = **460 Б/кадр**.
При 20 fps — **9.2 КБ/с**. Чанк core8 — 3.7 КБ, core40 — 18.4 КБ. Никакого сжатия не требуется.

### 3.6 Устойчивость

* Reconnect с экспоненциальным backoff; после реконнекта Unity шлёт `requestRange` от последнего валидного кадра.
* `revision` в `chunk`: инкрементируется при каждом `restart`. Чанки со старой ревизией отбрасываются.
* Python-сервер — один клиент за раз; второе подключение отклоняется с `error`.
* Heartbeat `status` раз в секунду — Unity рисует индикатор жизни.

---

## 4. Unity: донорский риг и Avatar

### 4.1 Процедурная генерация иерархии

По `hello`: создать `GameObject` на каждый сустав, `parent` по массиву `parents`, `localPosition = ArdyToUnity(neutral[i] − neutral[parent])`, `localRotation = identity`, `localScale = one`. Итог — GameObject-дерево, геометрически идентичное T-позе ARDY.

Живёт в сцене как `ArdyDonorRig` (компонент `ArdyDonor` хранит имена/индексы и кэш `Transform[]`).

### 4.2 Полная таблица маппинга cskel27 → HumanBodyBones

| idx | ARDY | HumanBodyBones | Обяз. |
|---|---|---|---|
| 0 | Hips | `Hips` | ✔ |
| 1 | Spine | `Spine` | ✔ |
| 2 | Spine1 | `Chest` | – |
| 3 | Spine2 | *(не мапится, см. §4.4)* | – |
| 4 | Spine3 | `UpperChest` | – |
| 5 | Neck | `Neck` | – |
| 6 | Head | `Head` | ✔ |
| 7 | RightShoulder | `RightShoulder` | – |
| 8 | RightArm | `RightUpperArm` | ✔ |
| 9 | RightForeArm | `RightLowerArm` | ✔ |
| 10 | RightHand | `RightHand` | ✔ |
| 11 | RightHandEnd | *(лист, не мапится)* | – |
| 12 | RightHandThumb1 | `RightThumbProximal` | – |
| 13 | LeftShoulder | `LeftShoulder` | – |
| 14 | LeftArm | `LeftUpperArm` | ✔ |
| 15 | LeftForeArm | `LeftLowerArm` | ✔ |
| 16 | LeftHand | `LeftHand` | ✔ |
| 17 | LeftHandEnd | *(лист, не мапится)* | – |
| 18 | LeftHandThumb1 | `LeftThumbProximal` | – |
| 19 | RightUpLeg | `RightUpperLeg` | ✔ |
| 20 | RightLeg | `RightLowerLeg` | ✔ |
| 21 | RightFoot | `RightFoot` | ✔ |
| 22 | RightToeBase | `RightToes` | – |
| 23 | LeftUpLeg | `LeftUpperLeg` | ✔ |
| 24 | LeftLeg | `LeftLowerLeg` | ✔ |
| 25 | LeftFoot | `LeftFoot` | ✔ |
| 26 | LeftToeBase | `LeftToes` | – |

Все 15 обязательных гуманоидных костей Unity присутствуют. **Строки `humanName` брать только как `HumanTrait.BoneName[(int)HumanBodyBones.LeftUpperArm]`** — хардкодить литералы нельзя, Unity сверяет их посимвольно.

### 4.3 Сборка аватара

```csharp
var desc = new HumanDescription {
    human    = humanBones,     // HumanBone[]: boneName (имя GameObject) → humanName
    skeleton = skeletonBones,  // SkeletonBone[]: ВСЕ трансформы, включая корневой GameObject
    upperArmTwist = 0.5f, lowerArmTwist = 0.5f,
    upperLegTwist = 0.5f, lowerLegTwist = 0.5f,
    armStretch = 0.05f, legStretch = 0.05f,
    feetSpacing = 0f, hasTranslationDoF = false
};
var avatar = AvatarBuilder.BuildHumanAvatar(donorRoot, desc);
if (!avatar.isValid) { /* лог + прерывание */ }
AssetDatabase.CreateAsset(avatar, "Assets/Ardy/Generated/ArdyCore27.asset");
```

Типовые причины `isValid == false` (проверять в этом порядке):
1. `skeleton[]` не содержит корневой GameObject → всегда включать `donorRoot`.
2. Имена в `human[].boneName` не совпадают с именами GameObject.
3. Поза не T-поза — у нас T-поза гарантирована, но только если риг сгенерирован до применения первого кадра. **Строить аватар строго на пустом риге.**
4. Ненулевые `localRotation` в момент сборки.

### 4.4 Проблема четырёх позвонков — честная оценка

У cskel27 четыре спинных кости (`Spine`, `Spine1`, `Spine2`, `Spine3`), у гуманоида Unity — три (`Spine`, `Chest`, `UpperChest`). Одна кость останется без маппинга, и её вращение **потеряется** при переходе в muscle-пространство.

Варианты, по возрастанию качества:
* **A (по умолчанию):** не мапить `Spine2`. Потеря — доля изгиба корпуса. Для локомоции и большинства промптов незаметна.
* **B:** перед построением human pose домножить поворот `Spine2` в `Spine3` (`Spine3_new = Spine2 · Spine3`) и обнулить `Spine2`. Изгиб сохраняется полностью, но перераспределяется — визуально ближе к оригиналу. Реализуется 5 строками в шаге обновления рига.
* **C:** бэйкать в generic-клип на донора и ретаргетить в рантайме — muscle-пространство вообще не участвует, теряется универсальность клипа.

Рекомендация: реализовать A, оставить B за флажком в настройках и сравнить на промптах с наклонами («bends down to pick something up»).

---

## 5. Live-предпросмотр в Editor

### 5.1 Сетевой слой и domain reload

Главная ловушка Editor-разработки: перекомпиляция C# убивает managed-объекты, но **не** закрывает сокеты и не останавливает `System.Threading.Thread` предсказуемо. Итог — «залипший» порт и фантомные потоки.

```csharp
[InitializeOnLoad]
static class ArdyEditorService {
    static ArdyEditorService() {
        AssemblyReloadEvents.beforeAssemblyReload += Shutdown;
        EditorApplication.quitting += Shutdown;
        EditorApplication.playModeStateChanged += OnPlayModeChanged;
        EditorApplication.update += Pump;
    }
}
```

* Приём — фоновый поток на блокирующем `NetworkStream.Read`; готовые сообщения складываются в `ConcurrentQueue<ArdyMessage>`.
* Разбор и запись в трансформы — **только** в `EditorApplication.update` (Unity API не потокобезопасен).
* Никаких `[ExecuteInEditMode]`-компонентов ради тика: он не даёт гарантий частоты и требует объекта в сцене. Черновик предлагал «проверять очередь каждую секунду» — при 20 fps это отставание на 20 кадров.
* Repaint: `SceneView.RepaintAll()` не чаще, чем нужно (throttle до ~60 Гц) — иначе Editor начнёт «пилить» CPU.

### 5.2 Jitter buffer и интерполяция

Генерация чанками по 0.4–2.0 с, а Editor тикает на 60–144 Гц. Без буфера предпросмотр будет дёргаться.

* Буфер: держать не менее 1 чанка «в запасе», начинать воспроизведение с задержкой `genHorizon / fps` секунд.
* Воспроизведение по собственным часам (`EditorApplication.timeSinceStartup`), позиция = `t · fps`, дробная часть → `Quaternion.Slerp` между соседними кадрами, `Vector3.Lerp` для корня.
* При исчерпании буфера — заморозка на последнем кадре + индикатор «underrun», без экстраполяции (экстраполяция кватернионов даёт заметные рывки).

### 5.3 Ретаргет на целевого персонажа в реальном времени

```csharp
// один раз:
srcHandler = new HumanPoseHandler(donorAvatar,  donorRoot.transform);
dstHandler = new HumanPoseHandler(targetAvatar, targetRoot.transform);
// каждый кадр:
srcHandler.GetHumanPose(ref pose);   // donor → muscle space (нормализовано по росту)
dstHandler.SetHumanPose(ref pose);   // muscle space → любой Humanoid
```

`HumanPose.bodyPosition` нормализован масштабом аватара — поэтому разница в росте персонажей обрабатывается автоматически, и таз не проваливается/не парит. **Это ответ на «как попасть на готовые 3D-модели» — без единой строки ручной ретаргет-математики.** Работает и в Edit Mode, и в Play Mode.

Требование к целевому персонажу: `Animation Type = Humanoid` в импортере, валидный Avatar. Для generic-персонажей путь только через ручную таблицу костей — это отдельный, существенно более дорогой сценарий; в первую версию не берём.

### 5.4 Play Mode

Тот же `ArdyClient`, но тик из `MonoBehaviour.Update`, а сокет живёт в `RuntimeInitializeOnLoadMethod`. Полезно для теста поведения в билде; для авторинга анимаций достаточно Edit Mode.

---

## 6. Бэйк в AnimationClip

### 6.1 Буфер записи

`MotionBuffer` — плотный `NativeArray<float>`/`float[]`, индексируемый абсолютным номером кадра ARDY. Пишется из **всех** входящих чанков (не из того, что показано), обрабатывает `invalidate` усечением. 10 минут анимации при 20 fps = 12 000 кадров × 460 Б ≈ 5.5 МБ. Держать в памяти целиком — нормально.

### 6.2 Три пути бэйка — оценка

| Путь | Что получаем | Универсальность | Риск | Вердикт |
|---|---|---|---|---|
| **P1.** `GameObjectRecorder` → `.anim` | Generic-клип с кривыми `localRotation`/`localPosition` донора | Только на риг с той же иерархией | Низкий | Фолбэк / отладка |
| **P2.** Ручные muscle-кривые через `HumanPoseHandler` + `AnimationUtility.SetEditorCurve` | Клип с биндингами на `Animator` (`RootT.*`, `RootQ.*`, 90+ muscle-каналов) | **Любой Humanoid** | Средний | **Основной** |
| **P3.** Экспорт FBX (`com.unity.formats.fbx`) → реимпорт с `animationType = Human` | Настоящий импортированный humanoid-клип | Любой Humanoid | Низкий, но тяжёлый | Гарантированный фолбэк |

**Важная поправка к черновику.** Утверждение «Unity автоматически трансформирует записанные углы в muscle space, потому что донор — Humanoid» неверно. `GameObjectRecorder` пишет то, на что подписан через `BindComponentsOfType<Transform>` — то есть трансформ-кривые. Полученный `.anim` останется generic-клипом; никакой автоматической muscle-конвертации при `SaveToClip` не происходит. Humanoid-клипы Unity штатно производит только импортёр моделей.

Поэтому P2 — единственный путь получить универсальный клип без экспорта в FBX, и именно его риск нужно снять первым спайком (S3).

### 6.3 P2 — детально

```
для каждого кадра t в диапазоне записи:
    применить кадр t к донор-ригу (localRotation/localPosition, без интерполяции)
    srcHandler.GetHumanPose(ref pose)
    записать pose.bodyPosition   → RootT.x/y/z
    записать pose.bodyRotation   → RootQ.x/y/z/w
    записать pose.muscles[i]     → HumanTrait.MuscleName[i]  (i = 0..HumanTrait.MuscleCount-1)
```

Каждый канал — свой `AnimationCurve`, ключ на кадр, время `t / fps`. Затем:

```csharp
var binding = EditorCurveBinding.FloatCurve("", typeof(Animator), channelName);
AnimationUtility.SetEditorCurve(clip, binding, curve);
clip.frameRate = fps;
```

Обязательные детали, каждая из которых иначе даёт видимый артефакт:

* **Непрерывность знака кватерниона `RootQ`.** Если `dot(q[t-1], q[t]) < 0` — инвертировать `q[t]`. Иначе интерполятор Unity прокрутит корень на 360° между кадрами. `GameObjectRecorder` делает это сам, ручная запись — нет. *(В черновике не упомянуто.)*
* **Тангенсы.** Ставить `AnimationUtility.SetKeyLeftTangentMode/SetKeyRightTangentMode` в `ClampedAuto`, иначе на резких кадрах будет перелёт (overshoot).
* **`AnimationClipSettings`** через `AnimationUtility.SetAnimationClipSettings`: `loopTime`, `loopBlend`, `keepOriginalOrientation`, `keepOriginalPositionY`, `cycleOffset`.
* **Имена muscle-каналов** — только `HumanTrait.MuscleName[i]`, не литералы.
* `AssetDatabase.CreateAsset(clip, path)` + `AssetDatabase.SaveAssets()`.

### 6.4 Root motion и in-place — поправка к черновику

Черновик предлагал «принудительно обнулять X и Z у Hips». Для гуманоидного клипа это неверно по двум причинам:

1. У humanoid-клипа корневое движение живёт в `RootT`, а не в позиции Hips. Обнуление Hips ломает соотношение таз↔корень.
2. Наивное обнуление XZ вызывает **проскальзывание стоп**: стопы были поставлены в мировых координатах, а корпус остался на месте.

Правильно — три режима на выбор пользователя:

* **`RootMotion` (по умолчанию):** ничего не трогаем. `Animator.applyRootMotion = true` воспроизведёт перемещение точно. Для геймплея это, как правило, то, что нужно.
* **`InPlace`:** обнулить кривые `RootT.x` и `RootT.z` (Y сохранить — прыжки/приседания), **и** одновременно обнулить накопленный yaw в `RootQ` (иначе персонаж будет вращаться на месте). Эквивалент импортёрского «Bake Into Pose».
* **`InPlacePreserveHeading`:** вычесть только поступательную составляющую, оставив разворот корпуса. Нужно для разворотов на месте.

Дополнительно: кнопка «Loop-friendly» — подобрать точку разреза по минимуму расстояния между позами (`sum |muscle[a] − muscle[b]|`) и выровнять первый/последний кадр.

### 6.5 Частота

Бэйкать на нативной частоте модели (20 или 25), `clip.frameRate = fps`. Не ресемплить в 30/60 — Unity интерполирует кривые сам, и любая частота воспроизведения будет плавной. Ресемплинг только добавит ключей и ошибку.

---

## 7. Качество: то, что решает, «выглядит как анимация» или «выглядит как нейросеть»

### 7.1 Проскальзывание стоп

Три уровня, применяются кумулятивно:

1. **`post_process_motion` на стороне Python** (`ardy/postprocess.py:184`) — уже есть, использует `foot_contacts`, включён в демо чекбоксом. Для core-модели включать всегда; для G1 отключён самими авторами.
2. **Передача `foot_contacts` в Unity** — 4 булевых канала в каждом чанке. Дёшево, а без них третий уровень невозможен.
3. **IK-полировка после ретаргета на целевого персонажа.** Разница пропорций донора и цели даёт остаточный скейт даже при идеальном источнике. `com.unity.animation.rigging` 1.4.1 **уже установлен** в проекте — `TwoBoneIKConstraint` на каждую ногу, вес = сглаженный сигнал контакта, цель фиксируется в мировой позиции на время контакта.

### 7.2 Голова — против жёсткого оверрайда

Черновик предлагал принудительно доворачивать `HumanBodyBones.Head` через `Quaternion.LookRotation` по вектору таза. Это ломает всё, ради чего используется text-to-motion: промпты вида «looks over their shoulder», «glances left», «looks up at the sky» перестанут работать, а при поворотах голова будет «примерзать» к тазу с эффектом робота.

Правильно:
* Стабилизация — опция, выключенная по умолчанию.
* Вес 0…1, а не жёсткая замена: `head = Slerp(head, stabilized, weight)`.
* База — не `Hips.forward` (таз качается при ходьбе), а **сглаженный heading** (у ARDY он уже есть — `global_root_heading`, cos/sin, можно передавать в чанке).
* Ограничение по углу (`Quaternion.RotateTowards` с лимитом ~25°), чтобы стабилизация только гасила дрожь, а не переопределяла намерение.

### 7.3 Дрожание на стыках чанков

Автогрессия шьёт окна по истории; на стыках возможны микроскачки. Меры: увеличить `history_crop_length`, включить кроссфейд 2–3 кадра на границе чанка в `MotionBuffer`, и не ставить `diffusionSteps` слишком низко ради FPS.

---

## 8. Оценка API: статус и риск

Легенда риска: 🟢 подтверждено чтением кода / стабильный публичный API · 🟡 нужен спайк · 🔴 требует внешней зависимости или обходного пути.

### Python / ARDY

| API | Назначение | Риск |
|---|---|---|
| `load_model(name, device, checkpoints_dir)` | загрузка чекпоинта | 🟢 |
| `Ardy.autoregressive_step(...)` | шаг генерации с историей | 🟢 сигнатура прочитана (`ardy_model.py:710`) |
| `Ardy.__call__(...)` | одноразовая генерация | 🟢 |
| `motion_rep.inverse(feats, is_normalized)` | фичи → повороты/позиции | 🟢 |
| `motion_rep.normalize/unnormalize` | нормализация | 🟢 |
| `post_process_motion(...)` | анти-скейт | 🟢 |
| `skeleton.neutral_joints`, `.joint_parents`, `.bone_order_names` | описание скелета | 🟢 |
| `save_bvh_file(...)` | BVH-экспорт | 🟢 (помнить `scale_to_cm`) |
| `CachedTextEncoder` | кэш эмбеддингов | 🟢 |
| `compute_window_num_frames` | бюджет окна | 🟡 живёт в `scripts/`, не в пакете — при headless-режиме скопировать |
| `InteractiveTimelineDemo._generate_step` | точка хука фазы 1 | 🟡 приватный метод демо, ломается при обновлении upstream |
| Сессия без viser-клиента | headless | 🔴 невозможно в демо (§0.6) → фаза 2 |
| Время шага < 0.4 с | реальное время | 🟡 **измерить (S0)** |
| TensorRT-путь | ускорение | 🟡 первая компиляция «несколько минут», нужен драйвер ≥525 |

### Unity / C#

| API | Назначение | Риск |
|---|---|---|
| `TcpClient` / `NetworkStream` / `ConcurrentQueue` | транспорт | 🟢 |
| `[InitializeOnLoad]`, `EditorApplication.update`, `AssemblyReloadEvents` | Editor-тик и жизненный цикл | 🟢 |
| `AvatarBuilder.BuildHumanAvatar(GameObject, HumanDescription)` | аватар из процедурного рига | 🟢 публичный, работает в Editor и рантайме; T-поза у нас гарантирована |
| `HumanTrait.BoneName[]`, `HumanTrait.MuscleName[]`, `HumanTrait.MuscleCount` | канонические имена | 🟢 |
| `HumanPoseHandler.GetHumanPose/SetHumanPose` | ретаргет donor→target | 🟢 |
| `AnimationUtility.SetEditorCurve` + `EditorCurveBinding.FloatCurve("", typeof(Animator), name)` | запись muscle-кривых | 🟡 **ключевое допущение всего пути P2 — спайк S3** |
| `AnimationUtility.SetAnimationClipSettings` | loop/root-настройки клипа | 🟢 |
| `UnityEditor.Animations.GameObjectRecorder` | фолбэк-бэйк | 🟢 — но `TakeSnapshot(float dt)` принимает **дельту времени**, а не абсолютное время; в черновике было `TakeSnapshot(frameIndex/20f)`, это дало бы экспоненциально растянутый клип |
| `AssetDatabase.CreateAsset/SaveAssets` | сохранение | 🟢 |
| `TwoBoneIKConstraint` (`com.unity.animation.rigging` 1.4.1) | IK ног | 🟢 пакет уже в манифесте |
| `com.unity.formats.fbx` | фолбэк P3 | 🔴 **не установлен**, добавляется отдельно |
| Импорт BVH | альтернативный путь | 🔴 Unity не умеет из коробки; сторонний парсер |

### Главная неизвестная

Есть ли у клипа с muscle-биндингами полноценный humanoid-статус (в частности `AnimationClip.humanMotion`) — определяется наличием этих кривых, и это надо подтвердить экспериментом, а не документацией. Спайк S3 отвечает на вопрос за один день; если ответ отрицательный — переходим на P3 (FBX Exporter), что стоит +1 зависимость и ~2 дня, но не меняет остальную архитектуру.

---

## 9. Этапы и критерии приёмки

| # | Этап | Результат | Критерий приёмки | Оценка |
|---|---|---|---|---|
| **S0** | Замер латентности | Таблица `core8` / `core40` × `diffusionSteps` × TRT on/off | Известно, укладывается ли шаг в 0.4 с | 0.5 д |
| **S1** | Golden-векторы | `golden_transform.json` + C#-тест | FK в Unity совпадает с `posed_joints` с ε=1e-4 | 1 д |
| **S2** | Аватар из процедурного рига | `ArdyCore27.asset` | `avatar.isValid == true`, ручная поза донора корректно переносится на готового Humanoid-персонажа | 1 д |
| **S3** | Muscle-кривые | Тестовый `.anim` из 3 ключей | Клип воспроизводится на **другом** Humanoid-персонаже через `Animator` | 1 д |
| **M1** | `ardy_bridge/` + `BridgeMixin` | Python шлёт `hello` + `chunk` | `nc`/тестовый скрипт видит корректный поток | 2 д |
| **M2** | `ArdyClient` + донор-риг | Донор шевелится в Scene View | Живой предпросмотр без Play Mode, без утечек при рекомпиляции | 2 д |
| **M3** | Live-ретаргет | Целевой персонаж повторяет донора | Стопы не проваливаются, рост не влияет | 1 д |
| **M4** | Управление из Unity | Промпт/play/pause/seek из Editor-окна | Смена промпта отражается в потоке < 1 с | 2 д |
| **M5** | Бэйк P2 | `.anim` в проекте | Клип работает на 3 разных персонажах | 3 д |
| **M6** | Root motion + in-place | 3 режима | Нет проскальзывания в in-place | 1.5 д |
| **M7** | Контакты + IK | Полировка ног | Скейт визуально устранён на «walk»/«run» | 2 д |
| **M8** | Headless `run_bridge.py` | Работа без браузера | Полный цикл из Unity при закрытом viser | 3 д |
| **M9** | Стабилизация, UX, доки | Editor-окно, пресеты, README | — | 2 д |

Итого ≈ **22 рабочих дня** до полнофункциональной версии; первый видимый результат (движущийся персонаж в Unity от промпта) — на M3, то есть ≈ 8-й день.

Критический путь: S3 → M5. Если S3 провалится — вставляется P3 (+2 д).

---

## 10. Ловушки, которые стоят дня отладки каждая

1. **Знак отражения.** `−Z` вместо `−X` даст «всё работает, но персонаж идёт спиной вперёд относительно мировых осей».
2. **Порядок компонент кватерниона.** `scipy.as_quat()` — `(x,y,z,w)`, `torch`/некоторые библиотеки — `(w,x,y,z)`. Явно фиксировать в протоколе и в тесте.
3. **`invalidate` при replan.** ARDY переписывает уже выданные кадры. Без обработки — рассинхрон буфера бэйка, проявляется только на записях > 1 чанка.
4. **Domain reload.** Незакрытый сокет и живой поток после рекомпиляции → «порт занят» и двойная запись в трансформы.
5. **`TakeSnapshot(dt)`, а не `TakeSnapshot(time)`.**
6. **Аватар строится на пустом риге.** Если применить кадр до `BuildHumanAvatar`, T-поза потеряна и аватар будет невалидным или кривым.
7. **Непрерывность знака `RootQ`** при ручной записи кривых.
8. **`scale_to_cm=True`** — дефолт `save_bvh_file`, даст персонажа ростом 175 метров.
9. **Литеральные имена muscle/bone.** Только через `HumanTrait`.
10. **Окно истории.** Без кропа истории длинные генерации деградируют в джиттер — это задокументировано самими авторами (`scripts/generate.py:104-110`).
11. **`purge_encoder_completely()` при `--offload`** после каждого шага — при частой смене промптов даёт перезагрузку энкодера; для интерактива запускать без `--offload`, если хватает VRAM.
12. **Unity API вне главного потока** — мгновенный краш или тихая порча состояния.

---

## 11. Что сознательно не делаем в первой версии

* Поддержка G1 (робот, не человек — гуманоидный аватар Unity к нему неприменим).
* Generic-персонажи без Humanoid Avatar (ручной bone-mapping — отдельная фича).
* Пальцы: cskel27 отдаёт только `Thumb1` и `HandEnd`. Полные кисти есть только у `somaskel77`, а модели под SOMA ещё не выпущены. Реалистичный план — статичная расслабленная поза кистей на целевом персонаже.
* Мимика, LookAt-таргеты, взаимодействие с объектами.
* Мультиперсонажные сцены (`num_samples > 1` даёт батч независимых сэмплов, не взаимодействующих персонажей).
