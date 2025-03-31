import GPUConnector from "../GPUConnector.js";

export default class Drawer extends GPUConnector {
  static UPDATE_FLAGS = Object.freeze({
    RESET: Number.MAX_SAFE_INTEGER,
    CURSOR_UPDATE: Number.MAX_SAFE_INTEGER - 1,
    IDLE: Number.MIN_SAFE_INTEGER,
    FORCE_RENDER: Number.MAX_SAFE_INTEGER - 2,
  });
  canvas;
  context;
  cursor = { x: -1, y: -1, z: -1 };
  updateFlag = Drawer.UPDATE_FLAGS.RESET;
  gpuData = Object.assign(this.gpuData, {
    ratio: 1,
    samplers: {},
    textures: {},
    renderPassDescriptor: {
      colorAttachments: [
        {
          clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
          view: null,
        },
      ],
      depthStencilAttachment: {
        view: null,
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    },
  });

  constructor(canvas = document.querySelector("canvas") || document.createElement("canvas")) {
    super();
    this.canvas = canvas;
    this.#setUpCanvasResizeListener();
  }

  async init(contextType) {
    this.context = this.canvas.getContext(contextType);
    if (!this.context) throw new Error("invalid context");

    if (contextType === "webgpu") await this.initGPU();
  }

  async initGPU() {
    await super.initGPU();
    this.context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "premultiplied",
      usage:
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  createSampler(
    name,
    samplerData = {
      magFilter: "linear",
      minFilter: "linear",
    }
  ) {
    this.setSampler(name, this.device.createSampler(samplerData));
  }

  getSampler(name) {
    return this.gpuData.samplers[name];
  }

  setSampler(name, sampler) {
    this.gpuData.samplers[name] = sampler;
  }

  async imageToBitmap(imageUrl) {
    return await createImageBitmap(await (await fetch(imageUrl)).blob());
  }

  async createTextureFromImage(imageUrl, name = "") {
    const bitmap = await this.imageToBitmap(imageUrl);
    this.setTexture(
      name || imageUrl,
      this.device.createTexture({
        size: [bitmap.width, bitmap.height, 1],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
      })
    );
    this.device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture: this.getTexture(name || imageUrl) },
      [bitmap.width, bitmap.height]
    );
  }

  #resizeCanvas() {
    const DEVICE_PIXEL_RATIO = window.devicePixelRatio;
    this.canvas.height = this.canvas.offsetHeight * DEVICE_PIXEL_RATIO;
    this.canvas.width = this.canvas.offsetWidth * DEVICE_PIXEL_RATIO;
    return [this.canvas.width, this.canvas.height];
  }

  #setUpCanvasResizeListener() {
    const work = () => {
      const [xLen, yLen] = this.#resizeCanvas();
      this.gpuData.ratio = xLen / yLen;
    };
    const wrappedResize = () => {
      document.removeEventListener("DOMContentLoaded", wrappedResize);
      work();
    };
    window.addEventListener("resize", work);
    document.addEventListener("DOMContentLoaded", wrappedResize);
  }

  setUpCursorListener() {
    this.canvas.addEventListener("mousemove", (e) => {
      if (this.updateFlag !== Drawer.UPDATE_FLAGS.IDLE) return;
      if (e.offsetX <= this.canvas.offsetWidth && e.offsetX >= 0) {
        this.cursor.y = e.offsetY / this.canvas.offsetHeight;
        this.updateFlag = Drawer.UPDATE_FLAGS.CURSOR_UPDATE;
      }
      if (e.offsetY <= this.canvas.offsetHeight && e.offsetY >= 0) {
        this.cursor.x = e.offsetX / this.canvas.offsetWidth;
        this.updateFlag = Drawer.UPDATE_FLAGS.CURSOR_UPDATE;
      }
    });
  }

  setUpKeyListener(keyPredicate, work) {
    document.addEventListener("keydown", (e) => {
      if (keyPredicate(e.key) && this.updateFlag === Drawer.UPDATE_FLAGS.IDLE)
        work(e.key);
    });
  }
}
