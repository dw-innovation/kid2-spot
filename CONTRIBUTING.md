# 🤝 Contributing to SPOT

Thank you for considering contributing to SPOT!  
We welcome help from developers, designers, journalists, mappers, and curious humans alike.

---

## 🧭 Where to Start

- Check the [Issues](https://github.com/dw-innovation/kid2-spot/issues) tab, or the issue tabs of the submodules.
- Look for labels like `good first issue`, `help wanted`, or `area/bundles`, `area/docs`.
- Read this file and the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

- One feature that requires the most community support is the tag bundle list that defines what OSM tags are usable in the application.
  → To help improve the list, please check [SPOT_OSM-tag-bundles.csv](../SPOT_OSM-tag-bundles.csv) and comment in the [pinned issue](https://github.com/dw-innovation/kid2-spot/issues/XXX)

---

## 🛠️ Local Setup

```bash
git clone --recurse-submodules https://github.com/dw-innovation/kid2-spot.git
cd kid2-spot
docker compose up --build
```

Each submodule also contains a `.env.example` you may need to configure.

---

## 💡 Ways to Contribute

- 📚 Improve documentation (README, usage tips)
- 🤖 Tweak prompts or data generation
- 🧪 Write tests
- 🎨 Improve the frontend UI
- 🌐 Translate content / improve accessibility
- 🧠 Suggest or curate new **OSM tag bundles**
  → Check [SPOT_OSM-tag-bundles.csv](../SPOT_OSM-tag-bundles.csv)  
  → Comment in the [pinned issue](https://github.com/dw-innovation/kid2-spot/issues/XXX)

---

## 💻 Code Style

- Use meaningful commit messages
- Keep PRs focused and small
- Document your code
- Add test coverage if possible

---

## ✅ Pull Request Process

1. Fork the repo and create your branch (`git checkout -b feature/thing`)
2. Commit your changes (`git commit -m 'Add new feature'`)
3. Push to the branch (`git push origin feature/thing`)
4. Open a Pull Request

All contributions must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 🙏 Thank You!

Whether it's reporting a bug, helping others, writing code, or improving documentation, we appreciate your support!