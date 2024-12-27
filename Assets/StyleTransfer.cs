using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;

public class StyleTransfer : MonoBehaviour
{
    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private RawImage contentRawImage;
    [SerializeField] private RawImage styleRawImage;
    [SerializeField] private RawImage outputRawImage;

    private IWorker worker;
    private Model runtimeModel;
    private RenderTexture outputRT;
    private Texture2D tempTexture;

    void Start() {
        // 모델 로드 및 실행기 초기화
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);

        outputRT = new RenderTexture(512, 512, 0, RenderTextureFormat.ARGB32);
        outputRawImage.texture = outputRT;
        tempTexture = new Texture2D(512, 512, TextureFormat.RGB24, false);

        // 모델 구조 자세히 출력
        Debug.Log("=== Model Structure ===");
        Debug.Log($"Number of inputs: {runtimeModel.inputs.Count}");
        for (int i = 0; i < runtimeModel.inputs.Count; i++) {
            var input = runtimeModel.inputs[i];
            Debug.Log($"Input {i}: Name={input.name}, Shape={input.shape}");
        }
        Debug.Log($"Number of outputs: {runtimeModel.outputs.Count}");
        for (int i = 0; i < runtimeModel.outputs.Count; i++) {
            var output = runtimeModel.outputs[i];
            Debug.Log($"Output {i}: Name={output}");
        }
    }

    private TensorFloat TextureToTensor(Texture sourceTexture, string debugName) {
        if (sourceTexture == null) {
            Debug.LogError($"{debugName} texture is null!");
            return null;
        }

        Debug.Log($"Converting {debugName} texture: {sourceTexture.width}x{sourceTexture.height}");

        var tempRT = RenderTexture.GetTemporary(512, 512, 0, RenderTextureFormat.ARGB32);
        Graphics.Blit(sourceTexture, tempRT);

        RenderTexture.active = tempRT;
        tempTexture.ReadPixels(new Rect(0, 0, 512, 512), 0, 0);
        tempTexture.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(tempRT);

        var tensorData = new float[512 * 512 * 3];
        var pixels = tempTexture.GetPixels();

        // 첫 몇 개의 픽셀값 출력
        Debug.Log($"{debugName} first pixel values: R={pixels[0].r}, G={pixels[0].g}, B={pixels[0].b}");

        for (int i = 0; i < pixels.Length; i++) {
            tensorData[i * 3] = pixels[i].r;
            tensorData[i * 3 + 1] = pixels[i].g;
            tensorData[i * 3 + 2] = pixels[i].b;
        }

        var tensor = new TensorFloat(new TensorShape(1, 3, 512, 512), tensorData);
        Debug.Log($"Created {debugName} tensor with shape: {tensor.shape}");
        return tensor;
    }

    public void ProcessImage() {
        Debug.Log("=== Starting Style Transfer ===");

        if (contentRawImage.texture == null || styleRawImage.texture == null) {
            Debug.LogError($"Content texture: {(contentRawImage.texture != null ? "exists" : "null")}, " +
                          $"Style texture: {(styleRawImage.texture != null ? "exists" : "null")}");
            return;
        }

        using var contentTensor = TextureToTensor(contentRawImage.texture, "Content");
        using var styleTensor = TextureToTensor(styleRawImage.texture, "Style");

        if (contentTensor == null || styleTensor == null) {
            Debug.LogError("Failed to create input tensors!");
            return;
        }

        try {
            // 모델의 첫 번째, 두 번째 입력으로 설정
            worker.SetInput("0", contentTensor);
            worker.SetInput("1", styleTensor);

            Debug.Log("Executing model...");
            worker.Execute();

            var output = worker.PeekOutput() as TensorFloat;
            Debug.Log($"Output tensor shape: {output.shape}");

            var outputPixels = new Color[512 * 512];
            var tensorSize = output.shape.length;
            Debug.Log($"Output tensor size: {tensorSize}");

            // 출력 데이터의 범위 확인
            float minValue = float.MaxValue;
            float maxValue = float.MinValue;

            var outputArray = new float[tensorSize];
            for (int i = 0; i < tensorSize; i++) {
                outputArray[i] = output[i];
                minValue = Mathf.Min(minValue, outputArray[i]);
                maxValue = Mathf.Max(maxValue, outputArray[i]);
            }

            Debug.Log($"Output value range: min={minValue}, max={maxValue}");

            for (int i = 0; i < 512 * 512; i++) {
                outputPixels[i] = new Color(
                    Mathf.Clamp01(outputArray[i * 3]),
                    Mathf.Clamp01(outputArray[i * 3 + 1]),
                    Mathf.Clamp01(outputArray[i * 3 + 2])
                );
            }

            var outputTex = new Texture2D(512, 512, TextureFormat.RGB24, false);
            outputTex.SetPixels(outputPixels);
            outputTex.Apply();

            Graphics.Blit(outputTex, outputRT);
            Debug.Log("Style transfer completed!");

            // 첫 번째 픽셀 값 확인
            var firstPixel = outputPixels[0];
            Debug.Log($"First output pixel: R={firstPixel.r}, G={firstPixel.g}, B={firstPixel.b}");

            Destroy(outputTex);
        } catch (System.Exception e) {
            Debug.LogError($"Error during style transfer: {e.Message}\n{e.StackTrace}");
        }
    }

    void OnDestroy() {
        worker?.Dispose();
        if (tempTexture != null)
            Destroy(tempTexture);
        if (outputRT != null)
            outputRT.Release();
    }
}