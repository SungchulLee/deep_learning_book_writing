# Optical Flow for Video Understanding

## Learning Objectives

By the end of this section, you will be able to:

- Understand the mathematical foundations of optical flow
- Implement classical flow algorithms (Lucas-Kanade, Farnebäck)
- Preprocess and visualize optical flow for neural networks
- Use flow as input to temporal stream networks
- Understand learned optical flow methods (FlowNet, RAFT)

## Mathematical Foundations

### Brightness Constancy Assumption

The fundamental assumption of optical flow is **brightness constancy**: a pixel's intensity remains constant as it moves between frames.

For intensity $I(x, y, t)$:

$$I(x, y, t) = I(x + u, y + v, t + \Delta t)$$

where $(u, v)$ is the displacement (flow) vector.

### Optical Flow Equation

Taylor expansion of the right side:

$$I(x + u, y + v, t + \Delta t) \approx I(x, y, t) + \frac{\partial I}{\partial x}u + \frac{\partial I}{\partial y}v + \frac{\partial I}{\partial t}\Delta t$$

Setting equal to brightness constancy and simplifying:

$$I_x u + I_y v + I_t = 0$$

or equivalently:

$$\nabla I \cdot \mathbf{v} + I_t = 0$$

where:
- $I_x, I_y$: Spatial gradients
- $I_t$: Temporal gradient
- $\mathbf{v} = (u, v)$: Flow vector

This is **one equation with two unknowns** — the aperture problem!

## Classical Algorithms

### Lucas-Kanade Method

Assumes flow is constant in a local neighborhood (window):

```python
import cv2
import numpy as np

def lucas_kanade_flow(img1: np.ndarray, 
                      img2: np.ndarray,
                      points: np.ndarray) -> tuple:
    """
    Lucas-Kanade sparse optical flow.
    
    Mathematical formulation:
    For each point p, solve least squares in window W:
        min Σ [I_x(q)u + I_y(q)v + I_t(q)]²
        q∈W
    
    Solution: (A^T A)^{-1} A^T b where
        A = [I_x(q), I_y(q)]  for q in W
        b = -I_t(q)
    
    Args:
        img1, img2: Grayscale images (H, W)
        points: Points to track (N, 1, 2)
    
    Returns:
        new_points: Updated point positions
        status: 1 if flow found, 0 otherwise
    """
    # Parameters
    lk_params = dict(
        winSize=(21, 21),      # Window size for local assumption
        maxLevel=3,            # Pyramid levels
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Calculate flow
    new_points, status, error = cv2.calcOpticalFlowPyrLK(
        img1, img2, points, None, **lk_params
    )
    
    return new_points, status


def detect_and_track(video_frames: list) -> list:
    """
    Detect features and track through video.
    """
    # Detect features in first frame
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    
    gray0 = cv2.cvtColor(video_frames[0], cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(gray0, mask=None, **feature_params)
    
    tracks = [points.copy()]
    
    # Track through video
    for i in range(1, len(video_frames)):
        gray1 = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2GRAY)
        
        new_points, status = lucas_kanade_flow(
            tracks[-1], gray1, points
        )
        
        # Keep only tracked points
        good_new = new_points[status == 1]
        tracks.append(good_new)
        points = good_new.reshape(-1, 1, 2)
        gray0 = gray1
    
    return tracks
```

### Farnebäck Dense Flow

Approximates each neighborhood with a polynomial:

$$f(x) \approx x^T A x + b^T x + c$$

Displacement is estimated from polynomial coefficients:

```python
def farneback_dense_flow(img1: np.ndarray, 
                         img2: np.ndarray) -> np.ndarray:
    """
    Farnebäck dense optical flow.
    
    Approximates each neighborhood with quadratic polynomial,
    estimates displacement from polynomial coefficient changes.
    
    Args:
        img1, img2: Grayscale images (H, W) as uint8
    
    Returns:
        flow: Dense flow field (H, W, 2) where flow[y,x] = (u, v)
    """
    flow = cv2.calcOpticalFlowFarneback(
        img1, img2, 
        None,           # Output flow
        pyr_scale=0.5,  # Pyramid scale
        levels=3,       # Pyramid levels
        winsize=15,     # Window size
        iterations=3,   # Iterations per level
        poly_n=5,       # Polynomial degree
        poly_sigma=1.2, # Gaussian for polynomial
        flags=0
    )
    
    return flow  # (H, W, 2)


def compute_flow_for_video(video: np.ndarray) -> np.ndarray:
    """
    Compute dense flow for entire video.
    
    Args:
        video: RGB video (T, H, W, 3) with values [0, 255]
    
    Returns:
        flows: Flow fields (T-1, H, W, 2)
    """
    T = video.shape[0]
    flows = []
    
    for t in range(T - 1):
        # Convert to grayscale
        gray1 = cv2.cvtColor(video[t], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(video[t + 1], cv2.COLOR_RGB2GRAY)
        
        # Compute flow
        flow = farneback_dense_flow(gray1, gray2)
        flows.append(flow)
    
    return np.stack(flows, axis=0)
```

### Horn-Schunck Global Method

Adds smoothness regularization:

$$E = \int\int \left[(I_x u + I_y v + I_t)^2 + \lambda^2(|\nabla u|^2 + |\nabla v|^2)\right] dx\, dy$$

where $\lambda$ controls smoothness weight.

```python
def horn_schunck_flow(img1: np.ndarray, 
                      img2: np.ndarray,
                      alpha: float = 1.0,
                      iterations: int = 100) -> np.ndarray:
    """
    Horn-Schunck optical flow (simplified implementation).
    
    Minimizes energy functional with:
    - Data term: brightness constancy
    - Smoothness term: flow should be smooth
    
    Solved iteratively with Gauss-Seidel.
    """
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    
    # Compute gradients
    Ix = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3)
    It = img2 - img1
    
    # Initialize flow
    u = np.zeros_like(img1)
    v = np.zeros_like(img1)
    
    # Averaging kernel
    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6,    0, 1/6],
                       [1/12, 1/6, 1/12]])
    
    # Iterative solution
    for _ in range(iterations):
        # Average of neighbors
        u_avg = cv2.filter2D(u, -1, kernel)
        v_avg = cv2.filter2D(v, -1, kernel)
        
        # Update
        denominator = alpha**2 + Ix**2 + Iy**2
        P = Ix * u_avg + Iy * v_avg + It
        
        u = u_avg - Ix * P / denominator
        v = v_avg - Iy * P / denominator
    
    return np.stack([u, v], axis=-1)
```

## Flow Visualization

### HSV Color Coding

Standard visualization encodes direction as hue, magnitude as saturation/value:

```python
def flow_to_rgb(flow: np.ndarray, 
                max_flow: float = None) -> np.ndarray:
    """
    Convert optical flow to RGB visualization.
    
    Color wheel encoding:
    - Hue: Flow direction (angle)
    - Saturation: Maximum
    - Value: Flow magnitude
    
    Args:
        flow: Optical flow (H, W, 2)
        max_flow: Maximum magnitude for normalization
    
    Returns:
        RGB image (H, W, 3) uint8
    """
    u, v = flow[..., 0], flow[..., 1]
    
    # Compute magnitude and angle
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    
    # Normalize magnitude
    if max_flow is None:
        max_flow = magnitude.max()
    magnitude = np.clip(magnitude / (max_flow + 1e-8), 0, 1)
    
    # Create HSV image
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = (magnitude * 255).astype(np.uint8)  # Value
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def visualize_flow_arrows(image: np.ndarray, 
                          flow: np.ndarray,
                          step: int = 16) -> np.ndarray:
    """
    Visualize flow as arrows overlaid on image.
    
    Args:
        image: RGB image (H, W, 3)
        flow: Optical flow (H, W, 2)
        step: Arrow spacing
    
    Returns:
        Image with flow arrows
    """
    H, W = flow.shape[:2]
    vis = image.copy()
    
    # Sample grid
    y, x = np.mgrid[step//2:H:step, step//2:W:step].reshape(2, -1).astype(int)
    
    # Get flow at sample points
    fx, fy = flow[y, x].T
    
    # Draw arrows
    for i in range(len(x)):
        pt1 = (x[i], y[i])
        pt2 = (int(x[i] + fx[i]), int(y[i] + fy[i]))
        cv2.arrowedLine(vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)
    
    return vis
```

## Preprocessing for Neural Networks

### Flow Normalization

```python
def normalize_flow_for_network(flow: np.ndarray,
                               bound: float = 20.0) -> np.ndarray:
    """
    Normalize optical flow for neural network input.
    
    Typical flow magnitudes are in [-20, 20] pixels.
    Normalize to [-1, 1] range.
    
    Args:
        flow: Raw flow (H, W, 2) or (T, H, W, 2)
        bound: Maximum expected flow magnitude
    
    Returns:
        Normalized flow with same shape
    """
    return np.clip(flow / bound, -1, 1)


def flow_to_tensor(flow: np.ndarray) -> torch.Tensor:
    """
    Convert flow to PyTorch tensor format.
    
    Args:
        flow: Flow array (H, W, 2) or (T, H, W, 2)
    
    Returns:
        Tensor (2, H, W) or (T, 2, H, W)
    """
    flow = torch.from_numpy(flow).float()
    
    if flow.dim() == 3:
        flow = flow.permute(2, 0, 1)  # (H, W, 2) -> (2, H, W)
    else:
        flow = flow.permute(0, 3, 1, 2)  # (T, H, W, 2) -> (T, 2, H, W)
    
    return flow
```

### Flow Stack for Temporal Stream

```python
def create_flow_stack(flows: np.ndarray, 
                      stack_length: int = 10) -> np.ndarray:
    """
    Create stacked flow input for temporal stream.
    
    The temporal stream expects L consecutive flows stacked
    along the channel dimension: (2*L, H, W)
    
    Args:
        flows: Flow sequence (T, H, W, 2)
        stack_length: Number of flows to stack (L)
    
    Returns:
        Stacked flows (2*L, H, W)
    """
    T = flows.shape[0]
    
    if T < stack_length:
        # Pad with zeros if not enough frames
        pad_size = stack_length - T
        padding = np.zeros((pad_size, *flows.shape[1:]))
        flows = np.concatenate([flows, padding], axis=0)
    
    # Take first stack_length flows
    flows = flows[:stack_length]  # (L, H, W, 2)
    
    # Stack along channel dimension
    stacked = flows.transpose(0, 3, 1, 2)  # (L, 2, H, W)
    stacked = stacked.reshape(-1, *stacked.shape[2:])  # (2*L, H, W)
    
    return stacked
```

## Learned Optical Flow

### FlowNet Architecture

```python
class FlowNetSimple(nn.Module):
    """
    FlowNet-S: Simple encoder-decoder for optical flow.
    (Dosovitskiy et al., 2015)
    
    Learns to predict optical flow from pairs of images.
    End-to-end trainable, faster than classical methods.
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoder: contracting path
        self.conv1 = nn.Conv2d(6, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        
        # Decoder: expanding path with skip connections
        self.deconv5 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(512 + 512, 256, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256 + 256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1)
        
        # Flow prediction at each scale
        self.predict_flow6 = nn.Conv2d(1024, 2, 3, padding=1)
        self.predict_flow5 = nn.Conv2d(512, 2, 3, padding=1)
        self.predict_flow4 = nn.Conv2d(256, 2, 3, padding=1)
        self.predict_flow3 = nn.Conv2d(128, 2, 3, padding=1)
        self.predict_flow2 = nn.Conv2d(64, 2, 3, padding=1)
        
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, img1: torch.Tensor, 
                img2: torch.Tensor) -> torch.Tensor:
        """
        Predict flow from image pair.
        
        Args:
            img1, img2: Images (B, 3, H, W)
        
        Returns:
            flow: Predicted flow (B, 2, H, W)
        """
        # Concatenate images
        x = torch.cat([img1, img2], dim=1)  # (B, 6, H, W)
        
        # Encoder
        conv1 = self.leaky_relu(self.conv1(x))
        conv2 = self.leaky_relu(self.conv2(conv1))
        conv3 = self.leaky_relu(self.conv3(conv2))
        conv4 = self.leaky_relu(self.conv4(conv3))
        conv5 = self.leaky_relu(self.conv5(conv4))
        conv6 = self.leaky_relu(self.conv6(conv5))
        
        # Decoder with skip connections
        flow6 = self.predict_flow6(conv6)
        
        deconv5 = self.leaky_relu(self.deconv5(conv6))
        concat5 = torch.cat([deconv5, conv5], dim=1)
        flow5 = self.predict_flow5(concat5[:, :512])
        
        deconv4 = self.leaky_relu(self.deconv4(concat5))
        concat4 = torch.cat([deconv4, conv4[:, :256]], dim=1)
        flow4 = self.predict_flow4(concat4[:, :256])
        
        deconv3 = self.leaky_relu(self.deconv3(concat4))
        concat3 = torch.cat([deconv3, conv3[:, :128]], dim=1)
        flow3 = self.predict_flow3(concat3[:, :128])
        
        deconv2 = self.leaky_relu(self.deconv2(concat3))
        flow2 = self.predict_flow2(deconv2)
        
        # Upsample to input resolution
        flow = F.interpolate(flow2, scale_factor=4, mode='bilinear', 
                            align_corners=False)
        
        return flow
```

### RAFT: Recurrent All-Pairs Field Transforms

```python
class RAFT(nn.Module):
    """
    RAFT: Recurrent All-Pairs Field Transforms (Teed & Deng, 2020)
    
    State-of-the-art learned optical flow:
    1. Feature extraction
    2. 4D correlation volume
    3. Iterative updates with GRU
    """
    
    def __init__(self, hidden_dim=128, context_dim=128, corr_levels=4):
        super().__init__()
        
        # Feature encoder
        self.fnet = FeatureEncoder(output_dim=256)
        
        # Context encoder
        self.cnet = ContextEncoder(output_dim=256)
        
        # Update block with GRU
        self.update_block = UpdateBlock(hidden_dim, context_dim)
        
        self.corr_levels = corr_levels
    
    def forward(self, img1, img2, iters=12):
        """
        Iteratively refine flow estimate.
        
        Args:
            img1, img2: Input images (B, 3, H, W)
            iters: Number of refinement iterations
        
        Returns:
            flow_predictions: List of flow estimates at each iteration
        """
        # Extract features
        fmap1 = self.fnet(img1)  # (B, 256, H/8, W/8)
        fmap2 = self.fnet(img2)
        
        # Build correlation volume
        corr = self.build_correlation_volume(fmap1, fmap2)
        
        # Context features
        cnet = self.cnet(img1)
        net, inp = torch.split(cnet, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        
        # Initialize flow
        coords = self.initialize_flow(img1)
        
        flow_predictions = []
        for _ in range(iters):
            # Lookup correlation
            corr_features = self.lookup_correlation(corr, coords)
            
            # Update
            net, delta_flow = self.update_block(net, inp, corr_features)
            coords = coords + delta_flow
            
            # Upsample and store
            flow_up = self.upsample_flow(coords - self.initialize_flow(img1))
            flow_predictions.append(flow_up)
        
        return flow_predictions
```

## Practical Considerations

### Computational Cost

| Method | Speed (fps) | Quality | GPU Required |
|--------|------------|---------|--------------|
| Lucas-Kanade | 30+ | Sparse | No |
| Farnebäck | 10-20 | Medium | No |
| TV-L1 | 1-5 | High | No |
| FlowNet-S | 100+ | Medium | Yes |
| FlowNet2 | 8-30 | High | Yes |
| RAFT | 5-10 | Best | Yes |

### Precomputing vs Online Computation

**Precompute when:**
- Training temporal stream networks
- Flow quality matters more than latency
- Storage is available

**Compute online when:**
- Real-time applications
- Storage constrained
- Can use fast approximations

## Summary

| Aspect | Key Points |
|--------|------------|
| **Brightness Constancy** | Fundamental assumption: $I_x u + I_y v + I_t = 0$ |
| **Lucas-Kanade** | Sparse, assumes local constant flow |
| **Farnebäck** | Dense, polynomial approximation |
| **Horn-Schunck** | Global, smoothness regularization |
| **Learned Methods** | End-to-end, faster, better quality |
| **For Neural Nets** | Normalize to [-1, 1], stack L flows |

## Next Steps

- **CNN-LSTM Models**: Use flow features with recurrent networks
- **Action Recognition**: Complete pipeline with flow
- **Self-Supervised Flow**: Learning flow without ground truth
