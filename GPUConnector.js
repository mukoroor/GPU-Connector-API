export default class GPUConnector {
  gpuData = {
    DEVICE: null,
    buffers: {},
    bindGroups: [],
    bindGroupLayouts: [],
    shaders: {},
    pipelines: {},
  };

  async initGPU(descriptor = {}) {
    if (!navigator.gpu) throw Error("WebGPU not supported.");

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw Error(`Couldn't request WebGPU ADAPTER.`);

    const device = await adapter.requestDevice(descriptor);
    this.device = device;
  }

  addBindGroup(bindGroup) {
    this.gpuData.bindGroups.push(bindGroup);
  }

  setPipeline(name, pipeline) {
    this.gpuData.pipelines[name] = pipeline;
  }

  getPipeline(name) {
    return this.gpuData.pipelines[name];
  }

  createShader(name, code) {
    this.setShader(name, this.device.createShaderModule({ label: name, code }));
  }

  getShader(name) {
    return this.gpuData.shaders[name];
  }

  setShader(name, shader) {
    this.gpuData.shaders[name] = shader;
  }

  allocateBuffer(size, usage, label=undefined) {
    if (this.device.limits.maxStorageBufferBindingSize < size)
      throw new Error("Buffer to Large");
    
    const buffer = this.device.createBuffer({ label, size, usage });
    return buffer;
  }

  allocateAndCacheBuffer(name, size, usage) {
    this.setBuffer(name, this.allocateBuffer(size, usage, name));
  }

  createCachedBuffer(name, data, usage, TypedArrayConstructor=undefined) {
    if (TypedArrayConstructor) data = TypedArrayConstructor(data);
    this.allocateAndCacheBuffer(name, data.byteLength, usage);
    this.writeCachedBuffer1to1(name, data);
    return this.getBuffer(name);
  }


  createBuffer(name, data, usage, TypedArrayConstructor=undefined) {
    if (TypedArrayConstructor) data = TypedArrayConstructor(data);
    const buffer = this.allocateBuffer(name, data.byteLength, usage);
    this.writeBuffer1to1(buffer, data);
    return buffer;
  }

  writeBuffer(buffer, bufferOffset, data, dataOffset, dataSize, TypedArrayConstructor=undefined) {
    if (TypedArrayConstructor) data = TypedArrayConstructor(data);
    this.device.queue.writeBuffer(
      buffer,
      bufferOffset,
      data,
      dataOffset,
      dataSize
    );
  }

  writeCachedBuffer(name, bufferOffset, data, dataOffset, dataSize, TypedArrayConstructor=undefined) {
    this.device.queue.writeBuffer(
      this.getBuffer(name),
      bufferOffset,
      data,
      dataOffset,
      dataSize,
      TypedArrayConstructor
    );
  }


  writeBuffer1to1(buffer, data, TypedArrayConstructor=undefined) {
    this.writeBuffer(buffer, 0, data, 0, data.length, TypedArrayConstructor);
  }

  writeCachedBuffer1to1(name, data, TypedArrayConstructor=undefined) {
    this.writeCachedBuffer(name, 0, data, 0, data.length, TypedArrayConstructor);
  }

  async copyBuffer(
    name,
    offset = 0,
    size,
    commandEncoder = undefined,
    end = false
  ) {
    const coercedCommandEncoder = commandEncoder || this.device.createCommandEncoder();

    const sourceBuff = this.getBuffer(name);
    size = size || sourceBuff.size;
    const copyName = `${name}_copy`;
    const destBuff = this.allocateBuffer(
      copyName,
      size,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    );

    coercedCommandEncoder.copyBufferToBuffer(
      sourceBuff,
      offset,
      destBuff,
      0,
      size
    );

    if (end || commandEncoder === undefined) {
      this.submitCommandEncoder(coercedCommandEncoder);
      await this.waitForDevice();
    }
  
    return destBuff;
  }

  async copyBuffers(perBufferParams, commandEncoder = undefined) {
    const coercedCommandEncoder = commandEncoder || this.device.createCommandEncoder();
    
    return await Promise.all(perBufferParams.map(({name, offset, size}, i) => this.copyBuffer(name, offset, size, coercedCommandEncoder, i === perBufferParams.length - 1)))
  }

  async printBuffer(name, TypedArrayConstructor, offset = 0, size, commandEncoder = undefined) {
    const copy = this.copyBuffer(name, size, offset, commandEncoder);  
    const data = await this.mapBuffer(copy, TypedArrayConstructor);
    
    console.log(`Buffer "${name}":`, data);
  }

  async printBuffers(perBufferParams, commandEncoder = undefined) {
    const copies = this.copyBuffers(perBufferParams, commandEncoder);  
    const data = await Promise.all((await copies).map((copy, i) => this.mapBuffer(copy, perBufferParams[i].TypedArrayConstructor)));
    
    data.forEach((_, i) => console.log(`Buffer "${perBufferParams[i].name}":`, data));
  }

  submitCommandEncoder(commandEncoder) {
    const commandBuffer = commandEncoder.finish();
    const start = performance.now();
    this.device.queue.submit([commandBuffer]);
    return start;
  }
  
  async mapBuffer(buff, TypedArrConstructor, offset, size) {
    await buff.mapAsync(GPUMapMode.READ, offset, size);
    const res = new TypedArrConstructor(buff.getMappedRange(offset, size));
    // buff.unmap();
    return res;
  }

  // async mapBuffers(buffers, TypedArrConstructor, offset, size) {
  //   await buff.mapAsync(GPUMapMode.READ, offset, size);
  //   const res = new TypedArrConstructor(buff.getMappedRange(offset, size));
  //   // buff.unmap();
  //   return res;
  // }

  getBuffer(name) {
    return this.gpuData.buffers[name];
  }

  setBuffer(name, buffer) {
    this.gpuData.buffers[name] = buffer;
  }

  setTexture(name, texture) {
    this.gpuData.textures[name] = texture;
  }

  getTexture(name) {
    return this.gpuData.textures[name];
  }

  set device(newDevice) {
    this.gpuData.DEVICE = newDevice;
  }

  get device() {
    return this.gpuData.DEVICE;
  }

  static waitForAnimationFrame(work) {
    return new Promise((resolve, reject) => {
      // Request the next animation frame and resolve the promise when it is called
      const refreshId = requestAnimationFrame(async (timeStamp) => {
        try {
          const result = await work(timeStamp);
          resolve(result || refreshId);
        } catch (e) {
          console.log(e);
          reject();
        }
      });
    });
  }

  async waitForDevice(){
    await this.device.queue.onSubmittedWorkDone();
  }
}
