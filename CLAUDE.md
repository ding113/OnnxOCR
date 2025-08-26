你正在开发一个onnxOCR 的 Rust服务端。这个服务端应该能够被使用Docker部署，暴露5005端口，在多路多核CPU的场景下，尽量榨干本地算力，实现CPU推理最大化。同时提供简单的Web UI和标准API接口（无鉴权），提供高性能、高安全、高可用的OCR服务端组件。

作为参考，一个示例的onnxocr模型文件位于resources\onnxocr\models\ppocrv5，你可以随时
复制出来进行测试。在resources\onnxocr里有一个用Python写的类似项目，但是其中对于并发、性能、安全、接口、前端、部署等部分的实现极为粗糙。因此你不应该照抄它的实现。只需要借鉴其中关于模型适配的经验。你务必全程ultrathink。

常见的OCR接口包括：

Request
curl -X POST http://localhost:5005/ocr \  
-H "Content-Type: application/json" \  
-d '{"image": "base64_encoded_image_data"}'  
Response
{  
  "processing_time": 0.456,  
  "results": [  
    {  
      "text": "Name",  
      "confidence": 0.9999361634254456,  
      "bounding_box": [[4.0, 8.0], [31.0, 8.0], [31.0, 24.0], [4.0, 24.0]]  
    },  
    {  
      "text": "Header",  
      "confidence": 0.9998759031295776,  
      "bounding_box": [[233.0, 7.0], [258.0, 7.0], [258.0, 23.0], [233.0, 23.0]]  
    }  
  ]  
}  


----

| Request Method | Request Body |
| ----------- | ----------- |
| POST        | file * (required) string($binary)  |
|             | force_ocr (option) boolean         |
|             | paginate_output (option) boolean   |
|             | output_format (option) string      |

---

Rust的onnx运行时可以参考以下文档：

https://github.com/pykeio/ort

ort is an open-source Rust binding for ONNX Runtime.

These docs are for the latest alpha version of ort, 2.0.0-rc.10. This version is production-ready (just not API stable) and we recommend new & existing projects use it.

ort makes it easy to deploy your machine learning models to production via ONNX Runtime , a hardware-accelerated inference engine. With ort + ONNX Runtime, you can run almost any ML model (including ResNet, YOLOv8, BERT, LLaMA) on almost any hardware, often far faster than PyTorch, and with the added bonus of Rust’s efficiency.

ONNX  is an interoperable neural network specification. Your ML framework of choice — PyTorch, TensorFlow, Keras, PaddlePaddle, etc. — turns your model into an ONNX graph comprised of basic operations like MatMul or Add. This graph can then be converted into a model in another framework, or inferenced directly with ONNX Runtime.

An example visual representation of an ONNX graph, showing how an input tensor flows through layers of convolution nodes.
Converting a neural network to a graph representation like ONNX opens the door to more optimizations and broader acceleration hardware support. ONNX Runtime can significantly improve the inference speed/latency of most models and enable acceleration with NVIDIA CUDA & TensorRT, Intel OpenVINO, Qualcomm QNN, Huawei CANN, and much more.

ort is the Rust gateway to ONNX Runtime, allowing you to infer your ONNX models via an easy-to-use and ergonomic API. Many commercial, open-source, & research projects use ort in some pretty serious production scenarios to boost inference performance:

Bloop’s semantic code search feature is powered by ort.
SurrealDB’s powerful SurrealQL query language supports calling ML models, including ONNX graphs through ort.
Google’s Magika file type detection library is powered by ort.
Wasmtime, an open-source WebAssembly runtime, supports ONNX inference for the WASI-NN standard  via ort.
rust-bert implements many ready-to-use NLP pipelines in Rust à la Hugging Face Transformers with both tch & ort backends.
Getting started
Add ort to your Cargo.toml
If you have a supported platform (and you probably do), installing ort couldn’t be any simpler! Just add it to your Cargo dependencies:


[dependencies]
ort = "=2.0.0-rc.10"
Convert your model
Your model will need to be converted to an ONNX graph before you can use it.

The awesome folks at Hugging Face have a guide  to export 🤗 Transformers models to ONNX with 🤗 Optimum.
For any PyTorch model: torch.onnx
For scikit-learn models: sklearn-onnx
For TensorFlow, Keras, TFlite, & TensorFlow.js: tf2onnx
For PaddlePaddle: Paddle2ONNX
Load your model
Once you’ve got a model, load it via ort by creating a Session:


use ort::session::{builder::GraphOptimizationLevel, Session};
 
let mut model = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .commit_from_file("yolov8m.onnx")?;
Perform inference
Preprocess your inputs, then run() the session to perform inference.


let outputs = model.run(ort::inputs!["image" => image]?)?;
let predictions = outputs["output0"].try_extract_array::<f32>()?;
...
Re-exports
pub use self::environment::init_from;
pub use self::environment::init;
pub use self::error::Error;
pub use self::error::ErrorCode;
pub use self::error::Result;
pub use ort_sys as sys;
Modules
adapter
An input adapter, allowing for loading many static inputs from disk at once.
api
compiler
editor
environment
An Environment is a process-global structure, under which Sessions are created.
error
execution_providers
ExecutionProviders provide hardware acceleration to Sessions.
io_binding
Enables binding of session inputs and/or outputs to pre-allocated memory.
logging
memory
Types for managing memory & device allocations.
metadata
operator
Contains traits for implementing custom operator domains & kernels.
session
Contains Session, the main interface used to inference ONNX models.
tensor
Traits and types related to Tensors.
trainingtraining
Provides Trainer, a simple interface for on-device training/fine-tuning.
util
value
Values are data containers used as inputs/outputs in ONNX Runtime graphs.
Macros
inputs
Construct the inputs to a session from an array or named map of values.
log
Logs a message to a given Logger.
ortsys
Constants
MINOR_VERSION
The minor version of ONNX Runtime used by this version of ort.
Traits
AsPointer
Trait to access raw pointers from safe types which wrap unsafe ort_sys types.
Functions
api
Returns a reference to the global ort_sys::OrtApi object.
info
Returns information about the build of ONNX Runtime used, including version, Git commit, and compile flags.
set_api
Sets the global ort_sys::OrtApi interface used by ort in order to use alternative backends, or a custom loading scheme.

----

Migrating from v1.x to v2
Environment no more
The Environment struct has been removed. Only one Environment was allowed per process, so it didn’t really make sense to have an environment as a struct.

To configure an Environment, you instead use the ort::init function, which returns the same EnvironmentBuilder as v1.x. Use commit() to then commit the environment.


ort::init()
    .with_execution_providers([CUDAExecutionProvider::default().build()])
    .commit()?;
commit() must be called before any sessions are created to take effect. Otherwise, a default environment will be created. The global environment can be updated afterward by calling commit() on another EnvironmentBuilder, however you’ll need to recreate sessions after comitting the new environment in order for them to use it.

Value specialization
The Value struct has been refactored into multiple strongly-typed structs: Tensor<T>, Map<K, V>, and Sequence<T>, and their type-erased variants: DynTensor, DynMap, and DynSequence.

Values returned by session inference are now DynValues, which behave exactly the same as Value in previous versions.

Tensors created from Rust, like via the new Tensor::new function, can be directly and infallibly extracted into its underlying data via extract_array (no try_):


let allocator = Allocator::new(&session, MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUInput)?)?;
let tensor = Tensor::<f32>::new(&allocator, [1, 128, 128, 3])?;
 
let array = tensor.extract_array();
// no need to specify type or handle errors - Tensor<f32> can only extract into an f32 ArrayView
You can still extract tensors, maps, or sequence values normally from a DynValue using try_extract_*:


let generated_tokens: ArrayViewD<f32> = outputs["output1"].try_extract_array()?;
DynValue can be downcast()ed to the more specialized types, like DynMap or Tensor<T>:


let tensor: Tensor<f32> = value.downcast()?;
let map: DynMap = value.downcast()?;
Similarly, a strongly-typed value like Tensor<T> can be upcast back into a DynValue or DynTensor.


let dyn_tensor: DynTensor = tensor.upcast();
let dyn_value: DynValue = tensor.into_dyn();
Tensor extraction directly returns an ArrayView
The new extract_array and try_extract_array functions return an ndarray::ArrayView directly, instead of putting it behind the old ort::value::Tensor<T> type (not to be confused with the new specialized value type). This means you don’t have to .view() on the result:


-let generated_tokens: Tensor<f32> = outputs["output1"].try_extract()?;
-let generated_tokens = generated_tokens.view();
+let generated_tokens: ArrayViewD<f32> = outputs["output1"].try_extract_array()?;
Full support for sequence & map values
You can now construct and extract Sequence/Map values.

Value views
You can now obtain a view of any Value via the new view() and view_mut() functions, which operate similar to ndarray’s own view system. These views can also now be passed into session inputs.

Mutable tensor extraction
You can extract a mutable ArrayViewMut or &mut [T] from a mutable reference to a tensor.


let (raw_shape, raw_data) = tensor.extract_tensor_mut();
Device-allocated tensors
You can now create a tensor on device memory with Tensor::new & an allocator:


let allocator = Allocator::new(&session, MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUInput)?)?;
let tensor = Tensor::<f32>::new(&allocator, [1, 128, 128, 3])?;
The data will be allocated by the device specified by the allocator. You can then use the new mutable tensor extraction to modify the tensor’s data.

Session creation
SessionBuilder::new(&environment) has been soft-replaced with Session::builder():


-// v1.x
-let session = SessionBuilder::new(&environment)?.with_model_from_file("model.onnx")?;
+// v2
+let session = Session::builder()?.commit_from_file("model.onnx")?;
SessionBuilder::with_model_* -> SessionBuilder::commit_*
The final SessionBuilder methods have been renamed for clarity.

SessionBuilder::with_model_from_file -> SessionBuilder::commit_from_file
SessionBuilder::with_model_from_memory -> SessionBuilder::commit_from_memory
SessionBuilder::with_model_from_memory_directly -> SessionBuilder::commit_from_memory_directly
SessionBuilder::with_model_downloaded -> SessionBuilder::commit_from_url
Session inputs
Tensor creation
You can now create input tensors from Arrays and ArrayViews. See the tensor value documentation for more information.

ort::inputs! macro
v2.0 makes the transition to the new input/output system easier by providing an inputs! macro. This new macro allows you to specify inputs either by position as they appear in the graph (like previous versions), or by name.

The ort::inputs! macro will painlessly convert compatible data types (see above) into the new inputs system.


-// v1.x
-let chunk_embeddings = text_encoder.run(&[CowArray::from(text_input_chunk.into_dyn())])?;
+// v2
+let chunk_embeddings = text_encoder.run(ort::inputs![text_input_chunk])?;
As mentioned, you can now also specify inputs by name using a map-like syntax. This is especially useful for graphs with optional inputs.


let noise_pred = unet.run(ort::inputs![
    "latents" => &latents,
    "timestep" => Tensor::from_array(([1], vec![t]))?,
    "encoder_hidden_states" => text_embeddings.view()
])?;
Tensor creation no longer requires the session’s allocator
In previous versions, Value::from_array took an allocator parameter. The allocator was only used because the string data contained in string tensors had to be cloned into ONNX Runtime-managed memory. However, 99% of users only ever use primitive tensors, so the extra parameter served little purpose. The new Tensor::from_array function now takes only an array, and the logic for converting string arrays has been moved to a new function, DynTensor::from_string_array.


-// v1.x
-let val = Value::from_array(session.allocator(), &array)?;
+// v2
+let val = Tensor::from_array(array)?;
Separate string tensor creation
As previously mentioned, the logic for creating string tensors has been moved from Value::from_array to DynTensor::from_string_array.

To use string tensors with ort::inputs!, you must create a Tensor using Tensor::from_string_array.


let array = ndarray::Array::from_shape_vec((1,), vec![document]).unwrap();
let outputs = session.run(ort::inputs![
    "input" => Tensor::from_string_array(array.view())?
])?;
Session outputs
New: Retrieve outputs by name
Just like how inputs can now be specified by name, you can now retrieve session outputs by name.


let l = outputs["latents"].try_extract_array::<f32>()?;
Execution providers
Execution provider structs with public fields have been replaced with builder pattern structs. See the API reference  and the execution providers reference for more information.


-// v1.x
-builder = builder.with_execution_providers(ExecutionProvider::DirectML(DirectMLExecutionProvider {
-    device_id: 1
-}))?;
+// v2
+builder = builder.with_execution_providers([
+    DirectMLExecutionProvider::default()
+        .with_device_id(1)
+        .build()
+])?;
Updated dependencies & features
ndarray 0.16
The ndarray dependency has been upgraded to 0.16. In order to convert tensors from ndarray, your application must update to ndarray 0.16 as well.

ndarray is now optional
The dependency on ndarray is now optional. If you previously used ort with default-features = false, you’ll need to add the ndarray feature to keep using ndarray integration.

Model Zoo structs have been removed
ONNX pushed a new Model Zoo structure that adds hundreds of different models. This is impractical to maintain, so the built-in structs have been removed.

You can still use Session::commit_from_url, it just now takes a URL string instead of a struct.

Changes to logging
Environment-level logging configuration (i.e. EnvironmentBuilder::with_log_level) has been removed because it could cause unnecessary confusion with our tracing integration.

Renamed types
The following types have been renamed with no other changes.

NdArrayExtensions -> ArrayExtensions
OrtResult, OrtError -> ort::Result, ort::Error
TensorDataToType -> ExtractTensorData
TensorElementDataType, IntoTensorElementDataType -> TensorElementType, IntoTensorElementType

Cargo features
✅ = default, ⚒️ = not default

✅ ndarray: Enables tensors to be created from/extracted to ndarray multi-dimensional arrays. We highly recommend this feature if you need to do a lot of complex pre/post-processing requiring multi-dimensional array access, but for something like an LLM, omitting this feature won’t require too much extra work but will save a fair amount of compile time.
✅ download-binaries: Downloads prebuilt binaries from pyke’s CDN service for supported platforms. Disabling this means you’ll need to compile ONNX Runtime from source yourself, and link ort to it.
✅ copy-dylibs: In case dynamic libraries are used (like with the CUDA execution provider), creates a symlink to them in the relevant places in the target folder to make compile-time dynamic linking work.
⚒️ half: Enables support for creating & extracting float16/bfloat16 tensors via the half crate. ONNX models that are converted to 16-bit precision will typically convert to/from 32-bit floats at the input/output, so you will likely never actually need to interact with a 16-bit tensor on the Rust side.
⚒️ num-complex: Enables support for creating & extracting complex32/complex64 tensors via the num-complex crate.
⚒️ load-dynamic: Enables runtime dynamic linking, which alleviates many of the troubles with compile-time dynamic linking and offers greater flexibility.
⚒️ alternative-backend: Disables linking to ONNX Runtime, allowing you to instead configure an alternative backend.
⚒️ fetch-models: Enables the SessionBuilder::commit_from_url method, allowing you to quickly download & run a model from a URL. This should only be used for quick testing.
Execution providers
Each execution provider is also gated behind a Cargo feature.

NVIDIA CUDA: cuda
NVIDIA TensorRT: tensorrt
Microsoft DirectML: directml
Apple CoreML: coreml
AMD ROCm: rocm
Intel OpenVINO: openvino
Intel oneDNN: onednn
XNNPACK: xnnpack
Qualcomm QNN: qnn
Huawei CANN: cann
Android NNAPI: nnapi
Apache TVM: tvm
Arm ACL: acl
ArmNN: armnn
AMD MIGraphX: migraphx
AMD Vitis AI: vitis
Rockchip RKNPU: rknpu
WebGPU: webgpu
Microsoft Azure: azure


Linking
ort provides its own builds of ONNX Runtime to make your experience as painless as possible, but in some cases, you’ll want to use a custom build of ONNX Runtime with ort. Luckily, we make this very easy by handling all of the linking configuration automagically. Just point ort to the output of ONNX Runtime’s build pipeline and it’ll Just Work™.

Static linking
Most ONNX Runtime compile configurations will support static linking - just run build.sh without the --build_shared_lib argument. You should prefer static linking if your execution providers support it, as it avoids many issues and follows de facto Rust practices. If you compile both static libraries and dynamic libraries, ort will prefer linking to the static libraries.

To direct ort to your statically built binaries, use the ORT_LIB_LOCATION environment variable when running cargo build. Point it to the location where the static libraries (.a/.lib files) are compiled to. This will typically be onnxruntime/build/<os>/<profile>. For example:


$ ORT_LIB_LOCATION=~/onnxruntime/build/Linux/Release cargo build
For iOS (or for other platforms if you are compiling multiple profiles at once), you’ll need to manually specify the profile with the ORT_LIB_PROFILE environment variable. If not specified, ort will prefer Release over RelWithDebInfo over MinSizeRel over Debug.

Dynamic linking
When it comes to dynamic linking, there are two options: load-dynamic, or standard compile-time dynamic linking. We recommend load-dynamic as it gives more control and is often far less troublesome to work with.

Runtime loading with load-dynamic
The load-dynamic Cargo feature solves a few of the issues with dynamic linking by loading the library at runtime rather than linking at compile time. This means that the path to the ONNX Runtime library can be configured at runtime, and the executable will not just completely fail to start if the binary couldn’t be found.

To use load-dynamic:

Enable the feature in Cargo.toml
Cargo.toml

[dependencies]
ort = { version = "2", features = [ "load-dynamic" ] }
Point ort to the dylib

fn main() -> anyhow::Result<()> {
    // Find our custom ONNX Runtime dylib path somehow
    // (i.e. resolving it from the root of our program's install folder)
    let dylib_path = crate::internal::find_onnxruntime_dylib()?;
    // The path should point to the `libonnxruntime` binary, which looks like:
    // - on Unix: /etc/.../libonnxruntime.so
    // - on Windows: C:\Program Files\...\onnxruntime.dll
 
    // Initialize ort with the path to the dylib. This **must** be called before any usage of `ort`!
    // `init_from` returns an `EnvironmentBuilder` which you can use to further configure the environment
    // before `.commit()`ing; see the Environment docs for more information on what you can configure.
    ort::init_from(dylib_path).commit()?;
 
    Ok(())
}
ORT_DYLIB_PATH is relative to the executable. Cargo examples and tests are compiled to a different directory than binary crates: target/<profile>/examples and target/<profile>/deps respectively. Keep this in mind if you’re going to use relative paths.
Compile-time dynamic linking
For compile-time dynamic linking, you’ll need to configure your environment in the exact same way as if you were statically linking.

Runtime dylib loading
Dylibs linked at compile-time need to be placed in a specific location for them to be found by the executable. For Windows, this is either somewhere on the PATH, or in the same folder as the executable. On macOS and Linux, they have to be placed somewhere in the LD_LIBRARY_PATH, or you can use rpath to configure the executable to search for dylibs in its parent folder. We’ve had the least issues with rpath, but YMMV.

To configure rpath, you’ll need to:

Enable rpath in Cargo.toml
Cargo.toml

[profile.dev]
rpath = true
 
[profile.release]
rpath = true
 
# do this for any other profiles
Configure the path in the linker args in .cargo/config.toml to be relative to the executable
~/.cargo/config.toml

[target.x86_64-unknown-linux-gnu]
rustflags = [ "-Clink-args=-Wl,-rpath,\\$ORIGIN" ]
 
# do this for any other Linux targets as well

Values
For ONNX Runtime, a value represents any type that can be given to/returned from a session or operator. Values come in three main types:

Tensors (multi-dimensional arrays). This is the most common type of Value.
Maps map a key type to a value type, similar to Rust’s HashMap<K, V>.
Sequences are homogenously-typed dynamically-sized lists, similar to Rust’s Vec<T>. The only values allowed in sequences are tensors, or maps of tensors.
Creating values
Creating tensors
Tensors can be created with Tensor::from_array from either:

an ndarray::Array, or
a tuple of (shape, data), where:
shape is one of Vec<I>, [I; N] or &[I], where I is i64 or usize, and
data is one of Vec<T> or Box<[T]>.

let tensor = Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 16, 16, 3)))?;
 
let tensor = Tensor::from_array(([1usize, 2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]))?;
The created tensor will take ownership of the passed data. See Creating views of external data to create temporary tensors referencing borrowed data.

Creating maps & sequences
Maps can be created  from any iterator yielding tuples of (K, V), where K and V are tensor element types.


let mut map = HashMap::<String, f32>::new();
map.insert("one".to_string(), 1.0);
map.insert("two".to_string(), 2.0);
map.insert("three".to_string(), 3.0);
 
let map = Map::<String, f32>::new(map)?;
Maps can also be created from 2 tensors , one containing keys and the other containing values:


let keys = Tensor::<i64>::from_array(([4], vec![0, 1, 2, 3]))?;
let values = Tensor::<f32>::from_array(([4], vec![1., 2., 3., 4.]))?;
 
let map = Map::new_kv(keys, values)?;
Sequences can be created  from any iterator yielding a Value subtype:


let tensor1 = Tensor::<f32>::new(&allocator, [1, 128, 128, 3])?;
let tensor2 = Tensor::<f32>::new(&allocator, [1, 224, 224, 3])?;
 
let sequence: Sequence<Tensor<f32>> = Sequence::new(vec![tensor1, tensor2])?;
Using values
Values can be used as an input in a session’s run function - either by value, by reference, or by view.


let latents = Tensor::<f32>::new(&allocator, [1, 128, 128, 3])?;
let text_embedding = Tensor::<f32>::new(&allocator, [1, 48, 256])?;
let timestep = Tensor::<f32>::new(&allocator, [1])?;
 
let outputs = session.run(ort::inputs![
    "timestep" => timestep,
    "latents" => &latents,
    "text_embedding" => text_embedding.view()
])?;
Extracting data
To access the underlying data of a value directly, the data must first be extracted.

Tensors can either extract to an ndarray::ArrayView via extract_array when the ndarray feature is enabled, or extract to a tuple via extract_tensor of (&Shape, &[T]) (where the second element is the slice of data contained within the tensor).


let array = ndarray::Array4::<f32>::ones((1, 16, 16, 3));
let tensor = TensorRef::from_array_view(&array)?;
 
let extracted: ArrayViewD<'_, f32> = tensor.extract_array();
let (tensor_shape, extracted_data): (&Shape, &[f32]) = tensor.extract_tensor();
Tensors and TensorRefMuts with non-string elements can also be mutably extracted with extract_array_mut and extract_tensor_mut. Mutating the returned types will directly update the data contained within the tensor.


let mut original_array = vec![1_i64, 2, 3, 4, 5];
{
	let mut tensor = TensorRefMut::from_array_view_mut(([original_array.len()], &mut *original_array))?;
	let (extracted_shape, extracted_data) = tensor.extract_tensor_mut();
	extracted_data[2] = 42;
}
assert_eq!(original_array, [1, 2, 42, 4, 5]);
Map and Sequence have Map::extract_map and Sequence::extract_sequence, which emit a HashMap<K, V> and a Vec of value views respectively. Unlike extract_tensor, these types cannot mutably extract their data, and always allocate on each extract call, making them more computationally expensive.

Session outputs return DynValues, which are values whose type is not known at compile time. In order to extract data from a DynValue, you must either downcast it to a strong type or use a corresponding try_extract_* method, which fails if the value’s type is not compatible:


let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
 
let Ok(tensor_output): ort::Result<ndarray::ArrayViewD<f32>> = outputs[0].try_extract_array() else {
    panic!("First output was not a Tensor<f32>!");
}
Views
A view (also called a ref) is functionally a borrowed variant of a value. There are also mutable views, which are equivalent to mutably borrowed values. Views are represented as separate structs so that they can be down/upcasted.

View types are suffixed with Ref or RefMut for shared/mutable variants respectively:

Tensors have DynTensorRef(Mut) and TensorRef(Mut).
Maps have DynMapRef(Mut) and MapRef(Mut).
Sequences have DynSequenceRef(Mut) and SequenceRef(Mut).
These views can be acquired with .view() or .view_mut() on a value type:


let my_tensor: ort::value::Tensor<f32> = Tensor::new(...)?;
 
let tensor_view: ort::value::TensorRef<'_, f32> = my_tensor.view();
Views act identically to a borrow of their type - TensorRef supports extract_tensor, TensorRefMut supports extract_tensor and extract_tensor_mut. The same is true for sequences & maps.

Creating views of external data
You can create TensorRefs and TensorRefMuts from views of external data, like an ndarray array, or a raw slice of data. These types act almost identically to a Tensor - you can extract them and pass them as session inputs - but as they do not take ownership of the data, they are bound to the input’s lifetime.


let original_data = Array4::<f32>::from_shape_vec(...);
let tensor_view = TensorRef::from_array_view(original_data.view())?;
 
let mut original_data = vec![...];
let tensor_view_mut = TensorRefMut::from_array_view_mut(([1, 3, 64, 64], &mut *original_data))?;
Dynamic values
Sessions in ort return a map of DynValues. These are values whose exact type is not known at compile time. You can determine a value’s type  via its .dtype() method.

You can also use fallible methods to extract data from this value - for example, DynValue::try_extract_tensor, which fails if the value is not a tensor. Often times though, you’ll want to reuse the same value which you are certain is a tensor - in which case, you can downcast the value.

Downcasting
Downcasting means to convert a dyn type like DynValue to stronger type like DynTensor. Downcasting can be performed using the .downcast() function on DynValue:


let value: ort::value::DynValue = outputs.remove("output0").unwrap();
 
let dyn_tensor: ort::value::DynTensor = value.downcast()?;
If value is not actually a tensor, the downcast() call will fail.

Stronger types
DynTensor means that the type is a tensor, but the element type is unknown. There are also DynSequences and DynMaps, which have the same meaning - the kind of value is known, but the element/key/value types are not.

The strongly typed variants of these types - Tensor<T>, Sequence<T>, and Map<K, V>, can be directly downcasted to, too:


let dyn_value: ort::value::DynValue = outputs.remove("output0").unwrap();
 
let f32_tensor: ort::value::Tensor<f32> = dyn_value.downcast()?;
If value is not a tensor, or if the element type of the value does not match what was requested (f32), the downcast() call will fail.

Stronger typed values have infallible variants of the .try_extract_* methods:


// We could try to extract a tensor directly from a `DynValue`...
let f32_array: ArrayViewD<f32> = dyn_value.try_extract_array()?;
 
// Or, we can first onvert it to a tensor, and then extract afterwards:
let tensor: ort::value::Tensor<f32> = dyn_value.downcast()?;
let f32_array = tensor.extract_array(); // no `?` required, this will never fail!
Upcasting
Upcasting means to convert a strongly-typed value type like Tensor<f32> to a weaker type like DynTensor or DynValue. This can be useful if you have code that stores values of different types, e.g. in a HashMap<String, DynValue>.

Strongly-typed value types like Tensor<f32> can be converted into a DynTensor using .upcast():


let dyn_tensor = f32_tensor.upcast();
// type is DynTensor
Tensor<f32> or DynTensor can be cast to a DynValue by using .into_dyn():


let dyn_value = f32_tensor.into_dyn();
// type is DynValue
Upcasting a value doesn’t change its underlying type; it just removes the specialization. You cannot, for example, upcast a Tensor<f32> to a DynValue and then downcast it to a Sequence; it’s still a Tensor<f32>, just contained in a different type.

Dyn views
Views also support down/upcasting via .downcast() & .into_dyn() (but not .upcast() at the moment).

You can also directly downcast a value to a stronger-typed view using .downcast_ref() and .downcast_mut():


let tensor_view: ort::value::TensorRef<'_, f32> = dyn_value.downcast_ref()?;
// is equivalent to
let tensor_view: ort::value::TensorRef<'_, f32> = dyn_value.view().downcast()?;
Conversion recap
DynValue represents a value that can be any type - tensor, sequence, or map. The type can be retrieved with .dtype().
DynTensor, DynMap, and DynSequence are values with known container types, but unknown element types.
Tensor<T>, Map<K, V>, and Sequence<T> are values with known container and element types.
Tensor<T> and co. can be converted from/to their dyn types using .downcast()/.upcast(), respectively.
Tensor<T>/DynTensor and co. can be converted to DynValues using .into_dyn().
An illustration of the relationship between value types as described above, used for visualization purposes.
Note that DynTensor cannot be downcast to Tensor<T>, but DynTensor can be upcast to DynValue with .into_dyn(), and then downcast to Tensor<T> with .downcast().

Type casting is computationally cheap; upcasts and .into_dyn() compile to a no-op.

I/O Binding
Often times when running a model with a non-CPU execution provider, you’ll find that the act of copying data between the device and CPU takes up a considerable amount of inference time.

In some cases, this I/O overhead is unavoidable — a causal language model, for example, must copy its sequence of input tokens to the GPU and copy the output probabilities back to the CPU to perform sampling on each run. In this case, there isn’t much room to optimize I/O. In other cases, though, you may have an input or output that does not need to be copied off of the device it is allocated on - i.e., if an input does not change between runs (such as a style embedding), or if an output is subsequently used directly as an input to another/the same model on the same device.

For these cases, ONNX Runtime provides I/O binding, an interface that allows you to manually specify which inputs/outputs reside on which device, and control when they are synchronized.

Creating
I/O binding is used via the IoBinding struct. IoBinding is created using the Session::create_binding method:


let mut binding = session.create_binding()?;
You’ll generally want to create one binding per “request”, as bound inputs/outputs only apply to individual instances of IoBinding.

Binding
Binding inputs
To bind an input, use IoBinding::bind_input. This will queue the input data to be copied to the device that session is allocated on.


let style_embedding: Tensor<f32> = Tensor::from_array(...)?;
 
binding.bind_input("style_embd", &style_embedding)?;
The data is not guaranteed to be synchronized immediately. The data will be fully synchronized once the I/O binding is run. To force synchronization, use IoBinding::synchronize_inputs.


binding.synchronize_inputs()?;
// all inputs are now synchronized
Binding an input represents a single copy at that moment in time. Any updates to style_embedding intentionally won’t take effect until you either call synchronize_inputs (which synchronizes all inputs), or re-bind style_embd (which will only synchronize style_embedding).

Binding outputs
Binding an output is similar; use IoBinding::bind_output, providing a value which the output will be placed into.


binding.bind_output("action", Tensor::<f32>::new(&Allocator::default(), [1, 32])?)?;
If you don’t know the output’s dimensions ahead of time, you can also simply bind to a device instead of providing a preallocated tensor:


let allocator = Allocator::default();
binding.bind_output_to_device("action", &allocator.memory_info())?;
In this example, when the I/O binding is run, the session output action will be placed into the same memory allocation provided in the bind_output call.

This means that subsequent runs will override the data in action. If you need to access a bound output’s data across runs (i.e. in a multithreading setting), the data needs to be copied to another buffer to avoid undefined behavior.

Outputs can be bound to any device — they can even stay on the EP device if you bind it to a tensor created with the session’s allocator (Tensor::new(session.allocator(), ...)). You can then access the pointer to device memory using Tensor::data_ptr.

If you do bind an output to the session’s device, it is not guaranteed to be synchronized after run, just like bind_input. You can force outputs to synchronize immediately using IoBinding::synchronize_outputs.

Running
To run a session using an I/O binding, you simply call the session’s run_binding() function with the created IoBinding:


let outputs = session.run_binding(&binding)?;
outputs provides the same interface as the outputs returned by Session::run, it just returns the outputs that you bound earlier.


// same `action` we allocated earlier in `bind_output`
let action: Tensor<f32> = outputs.remove("action").unwrap().downcast()?;
All together
Here is a more complete example of the I/O binding API in a scenario where I/O performance can be improved significantly. This example features a typical text-to-image diffusion pipeline, using a text encoder like CLIP to create the condition tensor and a UNet for diffusion.


let mut text_encoder = Session::builder()?
	.with_execution_providers([CUDAExecutionProvider::default().build()])?
	.commit_from_file("text_encoder.onnx")?;
let mut unet = Session::builder()?
	.with_execution_providers([CUDAExecutionProvider::default().build()])?
	.commit_from_file("unet.onnx")?;
 
let text_condition = {
    let mut binding = text_encoder.create_binding()?;
    binding.bind_input("tokens", &Tensor::<i64>::from_array((
        vec![1, 22],
        vec![49, 272, 503, 286, 1396, 353, 9653, 284, 1234, 287, 616, 2438, 11, 7926, 13, 3423, 338, 3362, 25, 12520, 238, 242]
    ))?)?;
    binding.bind_output_to_device("output0", &text_encoder.allocator().memory_info())?;
    text_encoder.run_binding(&binding)?.remove("output0").unwrap()
};
 
let input_allocator = Allocator::new(
	&unet,
	MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUInput)?
)?;
let mut latents = Tensor::<f32>::new(&input_allocator, [1, 4, 64, 64])?;
 
let mut io_binding = unet.create_binding()?;
io_binding.bind_input("condition", &text_condition)?;
 
let output_allocator = Allocator::new(
	&unet,
	MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUOutput)?
)?;
io_binding.bind_output("noise_pred", Tensor::<f32>::new(&output_allocator, [1, 4, 64, 64])?)?;
 
for _ in 0..20 {
	io_binding.bind_input("latents", &latents)?;
	let noise_pred = unet.run_binding(&io_binding)?.remove("noise_pred").unwrap();
 
	let mut latents = latents.extract_array_mut();
	latents += &noise_pred.try_extract_array::<f32>()?;
}
I/O binding provides 3 key performance boosts here:

Since we don’t use the text embeddings on the CPU, we can keep them on the GPU and avoid an expensive device-CPU-device copy.
Since the text condition tensor stays constant across each run of the UNet, we can use IoBinding to only copy it once.
Since the output tensor is always the same shape, we can pre-allocate the output in faster pinned memory and re-use the same allocation for each run.

Alternative backends
Since ONNX Runtime  is written in C++, linking troubles often arise when attempting to use it in a Rust project - especially with WASM. v2.0.0-rc.10 of ort introduced support for alternative backends — that is, ONNX executors that do not use ONNX Runtime.

As the Rust ML scene has evolved, many exciting new inference engines supporting ONNX models have popped up, like 🤗 Hugging Face’s candle, Burn , and tract. These libraries, being written in pure Rust (minus some GPU kernels) play much nicer when it comes to linking, and often support any platform Rust’s standard library does. They’re also, of course, memory safe and 🦀blazingly🔥fast🚀!

Internally, alternative backend implementations are simply glue code between these libraries and the ONNX Runtime C API. Because they implement the same API as ONNX Runtime, using them in ort is as simple as adding one line of code!

Using an alternative backend
Alternative backends are experimental, and are constantly changing and growing — use them at your own risk!

We may not be able to provide the same level of support for different backends as we do with ONNX Runtime.

Install the alternative backend
We’ll use ort-tract for this example.

Cargo.toml

[dependencies]
ort-tract = "0.1.0+0.21"
...
Enable the alternative-backend feature
This instructs ort to not try to download/link to ONNX Runtime.

Cargo.toml

[dependencies.ort]
version = "=2.0.0-rc.10"
default-features = false # Disables the `download-binaries` feature since we don't need it
features = [
    "alternative-backend"
]
Initialize the backend
Use ort::set_api to use the crate’s API implementation (replacing ort_tract with whichever backend crate you choose to use):


fn main() {
    // This should run as early in your application as possible - before you ever use `ort`!
    ort::set_api(ort_tract::api());
}
Done!
Be sure to check each backend’s docs page to see which APIs are and are not implemented.

Available backends
ort currently has the following backends:

ort-candle, based on 🤗 Hugging Face candle
🔷 Supports: CPU, CUDA (though not available via ort-candle right now), WebAssembly
⚠️ Limited operator support; though most transformer models have good support.
ort-tract, based on tract
🔷 Supports: CPU, WebAssembly
✅ Great operator support 
ort-candle
ort-candle is an alternative backend for ort based on 🤗 Hugging Face candle.

Supported APIs
✅ ort::init
🔷 ort::environment::EnvironmentBuilder
EnvironmentBuilder::commit
🔷 ort::memory::Allocator
Allocator::default
Allocator::memory_info
✅ ort::memory::MemoryInfo
🔷 ort::session::Session
Session::builder
Session::allocator
Session::run
Session::run_with_options
🔷 ort::session::builder::SessionBuilder
SessionBuilder::new
SessionBuilder::commit_from_file
SessionBuilder::commit_from_memory
SessionBuilder::commit_from_memory_directly
SessionBuilder::commit_from_url
✅ ort::value::DynValue, ort::value::DynValueRef, ort::value::DynValueRefMut
Only Tensor types are supported.
✅ ort::value::Tensor, TensorRef, TensorRefMut, etc.
✅ ort::value::ValueType
Usage
Install ort-candle
Cargo.toml

[dependencies]
ort-candle = "0.1.0+0.8"
...
Enable the alternative-backend feature
This instructs ort to not try to download/link to ONNX Runtime.

Cargo.toml

[dependencies.ort]
version = "=2.0.0-rc.10"
default-features = false # Disables the `download-binaries` feature since we don't need it
features = [
    "alternative-backend"
]
Initialize the backend
Use ort::set_api to use the crate’s API implementation.


fn main() {
    // This should run as early in your application as possible - before you ever use `ort`!
    ort::set_api(ort_candle::api());
}
Done!
ort-tract
ort-tract is an alternative backend for ort based on tract.

Supported APIs
✅ ort::init
🔷 ort::environment::EnvironmentBuilder
EnvironmentBuilder::commit
🔷 ort::memory::Allocator
Allocator::default
Allocator::memory_info
✅ ort::memory::MemoryInfo
🔷 ort::session::Session
Session::builder
Session::allocator
Session::run
Session::run_with_options
🔷 ort::session::builder::SessionBuilder
SessionBuilder::new
SessionBuilder::commit_from_file
SessionBuilder::commit_from_memory
SessionBuilder::commit_from_memory_directly
SessionBuilder::commit_from_url
SessionBuilder::with_optimization_level
✅ ort::value::DynValue, ort::value::DynValueRef, ort::value::DynValueRefMut
Only Tensor types are supported.
✅ ort::value::Tensor, TensorRef, TensorRefMut, etc.
✅ ort::value::ValueType
Usage
Install ort-tract
Cargo.toml

[dependencies]
ort-tract = "0.1.0+0.21"
...
Enable the alternative-backend feature
This instructs ort to not try to download/link to ONNX Runtime.

Cargo.toml

[dependencies.ort]
version = "=2.0.0-rc.10"
default-features = false # Disables the `download-binaries` feature since we don't need it
features = [
    "alternative-backend"
]
Initialize the backend
Use ort::set_api to use the crate’s API implementation.


fn main() {
    // This should run as early in your application as possible - before you ever use `ort`!
    ort::set_api(ort_tract::api());
}
Done!
Troubleshooting: Logging
ort hooks into ONNX Runtime to route its logging messages through the tracing crate. These logging messages can often provide more helpful information about specific failure modes than ort’s error messages alone.

To enable logging for ort, you need to set up a tracing subscriber in your application, such as tracing-subscriber. tracing-subscriber’s fmt subscriber logs readable (and quite pretty!) messages to the console. To set it up:

Add tracing-subscriber to your dependencies

[dependencies]
tracing-subscriber = { version = "0.3", features = [ "env-filter", "fmt" ] }
Initialize the subscriber in the main function

fn main() {
    tracing_subscriber::fmt::init();
}
Show debug messages from ort
The environment variable RUST_LOG configures filters for crates that use tracing; see tracing_subcriber::EnvFilter for more information.

Set RUST_LOG to ort=debug to see all debug messages from ort. (You can also set it to trace for more verbosity, or info, warn, or error for less.)


$env:RUST_LOG = 'ort=debug';
cargo run
Troubleshooting: Issues compiling/linking
The trait bound ort::value::Value: From<...> is not satisfied
An error like this might come up when attempting to upgrade from an earlier (1.x) version of ort to a more recent version:


error[E0277]: the trait bound `ort::value::Value: From<ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>>` is not satisfied
  --> src/main.rs:72:16
   |
72 |           let inputs = ort::inputs![
   |  ______________________^
73 | |             input1,
74 | |         ]?;
   | |_________^ the trait `From<ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>>` is not implemented for `ort::value::Value`, which is required by `ort::value::Value: TryFrom<ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>>`
   |
   = help: the following other types implement trait `From<T>`:
             `ort::value::Value` implements `From<ort::value::Value<DynTensorValueType>>`
             `ort::value::Value` implements `From<ort::value::Value<TensorValueType<T>>>`
   = note: required for `ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>` to implement `Into<ort::value::Value>`
   = note: required for `ort::value::Value` to implement `TryFrom<ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>>`
   = note: this error originates in the macro `ort::inputs` (in Nightly builds, run with -Z macro-backtrace for more info)
Recent versions of ort require ndarray 0.16, whereas older versions (and thus possibly your code) required 0.15. Since these versions are semver incompatible, Cargo treats the ndarray used by your crate and the ndarray used by ort as separate crates entirely; hence the contradictory error message.

To fix this, upgrade your ndarray dependency to 0.16; the new release features no breaking changes, although .into_shape() is deprecated; see ndarray’s release notes for more information.

Unresolved external symbol __std_*
If you encounter these errors when linking on Windows, make sure your Visual Studio 2022 installation is up to date; at least version 17.11 is required when using default pyke binaries.
Version mapping
Versions of ONNX Runtime used by ort
ort	ONNX Runtime
v2.0.0+	v1.22.0
v1.16.0-v1.16.2	v1.16.0
v1.15.0-v1.15.5	v1.15.1
v1.14.2-v1.14.8	v1.14.1
v1.14.0-v1.14.1	v1.14.0
v1.13.1-v1.13.3	v1.13.1
v1.13.0	v1.12.1
Supported ONNX opsets by ort version
Note that this only applies to the default ONNX Runtime backend.

ort	ONNX opset version	ONNX ML opset version
v2.0.0+	22	4
v1.16.0-v1.16.2	19	3
v1.15.0-v1.15.5	19	3
v1.14.0-v1.14.8	18	3
v1.13.0-v1.13.3	17	3
A note on SemVer
ort versions pre-2.0 were not SemVer compatible. From v2.0 onwards, breaking API changes are accompanied by a major version update.

Updates to the version of ONNX Runtime used by ort may occur on minor version updates, i.e. 2.0 ships with ONNX Runtime 1.22, but 2.1 may ship with 1.23. ONNX Runtime is generally forward compatible, but in case you require a specific version of ONNX Runtime, you should pin the minor version in your Cargo.toml using a tilde requirement :


[dependencies]
ort = { version = "~2.0", ... }