from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from numpy.linalg import norm

SZ = 20
PROVINCE_START = 1000

# Keep pinyin order identical to the original project so labels stay compatible.
PROVINCES = [
	"zh_cuan",
	"zh_e",
	"zh_gan",
	"zh_gan1",
	"zh_gui",
	"zh_gui1",
	"zh_hei",
	"zh_hu",
	"zh_ji",
	"zh_jin",
	"zh_jing",
	"zh_jl",
	"zh_liao",
	"zh_lu",
	"zh_meng",
	"zh_min",
	"zh_ning",
	"zh_qing",
	"zh_qiong",
	"zh_shan",
	"zh_su",
	"zh_sx",
	"zh_wan",
	"zh_xiang",
	"zh_xin",
	"zh_yu",
	"zh_yu1",
	"zh_yue",
	"zh_yun",
	"zh_zang",
	"zh_zhe",
]

VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class StatModel:
	def load(self, fn: str) -> None:
		"""从磁盘加载已训练好的 OpenCV 模型文件。"""
		self.model = self.model.load(fn)

	def save(self, fn: str) -> None:
		"""将当前模型参数保存到指定文件。"""
		self.model.save(fn)


class SVM(StatModel):
	def __init__(self, kernel: int = cv2.ml.SVM_RBF, c: float = 1.0, gamma: float = 0.5):
		"""初始化一个使用 RBF 核的 SVM 分类器。"""
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(c)
		self.model.setKernel(kernel)
		self.model.setType(cv2.ml.SVM_C_SVC)

	def train(self, samples: np.ndarray, responses: np.ndarray) -> None:
		"""使用特征向量和标签训练 SVM。"""
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

	def predict(self, samples: np.ndarray) -> np.ndarray:
		"""对输入特征做分类并返回预测标签。"""
		result = self.model.predict(samples)
		return result[1].ravel()


def deskew(img: np.ndarray) -> np.ndarray:
	"""对字符图做去倾斜矫正以降低书写偏斜影响。"""
	m = cv2.moments(img)
	if abs(m["mu02"]) < 1e-2:
		return img.copy()
	skew = m["mu11"] / m["mu02"]
	matrix = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
	return cv2.warpAffine(img, matrix, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def preprocess_hog(images: List[np.ndarray]) -> np.ndarray:
	"""将字符图批量提取为 HOG 特征向量。"""
	samples = []
	for img in images:
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bin_n = 16
		bins = np.int32(bin_n * ang / (2 * np.pi))

		bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
		mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)

		eps = 1e-7
		hist /= hist.sum() + eps
		hist = np.sqrt(hist)
		hist /= norm(hist) + eps
		samples.append(hist)

	return np.float32(samples)


def is_image_file(path: Path) -> bool:
	"""判断路径是否为支持的图片文件。"""
	return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXT


def get_class_dirs(root: Path, class_filter: Callable[[str], bool]) -> List[Path]:
	"""读取并返回满足类别过滤条件的子目录列表。"""
	if not root.exists():
		raise FileNotFoundError(f"Dataset root does not exist: {root}")
	class_dirs = [d for d in root.iterdir() if d.is_dir() and class_filter(d.name)]
	class_dirs.sort(key=lambda p: p.name)
	return class_dirs


def load_features_from_dir(
	dataset_dir: Path,
	label_from_class: Callable[[str], int],
	class_filter: Callable[[str], bool],
) -> Tuple[np.ndarray, np.ndarray]:
	"""从类别目录加载样本并转换为特征与标签数组。"""
	images: List[np.ndarray] = []
	labels: List[int] = []

	class_dirs = get_class_dirs(dataset_dir, class_filter)
	for class_dir in class_dirs:
		label = label_from_class(class_dir.name)
		files = [p for p in class_dir.iterdir() if is_image_file(p)]
		files.sort(key=lambda p: p.name)
		for file_path in files:
			img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
			if img is None:
				continue
			if img.shape[0] != SZ or img.shape[1] != SZ:
				img = cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_AREA)
			images.append(img)
			labels.append(label)

	if not images:
		return np.empty((0, 64), dtype=np.float32), np.empty((0,), dtype=np.int32)

	deskewed = list(map(deskew, images))
	feats = preprocess_hog(deskewed)
	lbls = np.array(labels, dtype=np.int32)
	return feats, lbls


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""计算预测结果相对真实标签的准确率。"""
	if y_true.size == 0:
		return 0.0
	return float((y_true == y_pred).sum() / y_true.size)


def evaluate_optional_split(
	svm: SVM,
	dataset_dir: Optional[Path],
	label_from_class: Callable[[str], int],
	class_filter: Callable[[str], bool],
	split_name: str,
) -> None:
	"""在可选验证/测试集上评估模型并打印准确率（目录格式与训练集一致）。"""
	if dataset_dir is None:
		print(f"  {split_name:<5} skipped (path not set)")
		return
	if not dataset_dir.exists():
		print(f"  {split_name:<5} skipped (path not found: {dataset_dir})")
		return

	x, y = load_features_from_dir(dataset_dir, label_from_class, class_filter)
	if x.size == 0:
		print(f"  {split_name:<5} skipped (no valid samples)")
		return
	pred = svm.predict(x)
	print(f"  {split_name:<5} samples={len(y):<6} acc={accuracy(y, pred):.4f}")


def build_stratified_kfold_indices(y: np.ndarray, k_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
	"""按类别分层构造 K 折索引，尽量保证每折类别分布均衡。"""
	if k_folds < 2:
		raise ValueError("k_folds must be >= 2")

	rng = np.random.default_rng(seed)
	all_indices = np.arange(len(y), dtype=np.int32)
	fold_buckets: List[List[int]] = [[] for _ in range(k_folds)]

	for label in np.unique(y):
		label_indices = np.where(y == label)[0].astype(np.int32)
		rng.shuffle(label_indices)
		for i, idx in enumerate(label_indices):
			fold_buckets[i % k_folds].append(int(idx))

	folds: List[Tuple[np.ndarray, np.ndarray]] = []
	for fold_id in range(k_folds):
		val_idx = np.array(fold_buckets[fold_id], dtype=np.int32)
		train_mask = np.ones(len(y), dtype=bool)
		train_mask[val_idx] = False
		train_idx = all_indices[train_mask]
		folds.append((train_idx, val_idx))

	return folds


def run_kfold_cv(
	x: np.ndarray,
	y: np.ndarray,
	k_folds: int,
	c: float,
	gamma: float,
	title: str,
	seed: int = 42,
) -> float:
	"""在给定训练集上执行 K 折交叉验证并返回平均准确率。"""
	folds = build_stratified_kfold_indices(y, k_folds, seed)
	fold_accs: List[float] = []

	print(f"\n[{title} - {k_folds}-Fold CV]")
	for i, (train_idx, val_idx) in enumerate(folds, start=1):
		if val_idx.size == 0:
			print(f"  fold={i:<2} skipped (no validation samples)")
			continue

		svm = SVM(c=c, gamma=gamma)
		svm.train(x[train_idx], y[train_idx])
		pred = svm.predict(x[val_idx])
		acc = accuracy(y[val_idx], pred)
		fold_accs.append(acc)
		print(
			f"  fold={i:<2} train={len(train_idx):<6} val={len(val_idx):<6} acc={acc:.4f}"
		)

	if not fold_accs:
		return 0.0

	mean_acc = float(np.mean(fold_accs))
	std_acc = float(np.std(fold_accs))
	print(f"  cv mean={mean_acc:.4f} std={std_acc:.4f}")
	return mean_acc


def find_best_svm_params_with_report(features: np.ndarray, labels: np.ndarray) -> Tuple[int, float, float]:
	"""
	搜索最优 SVM 参数并输出所有组合的准确率对比表。
	返回：best_kernel, best_c, best_gamma
	"""
	C_LIST = [0.1, 1, 10, 100]
	GAMMA_LIST = [0.01, 0.1, 0.5, 1, 10]
	BEST_KERNEL = cv2.ml.SVM_RBF

	if features.size == 0 or labels.size == 0 or len(features) < 2:
		print("=" * 80)
		print("SVM 参数组合 准确率对比表")
		print("=" * 80)
		print("样本不足，回退默认参数：kernel=RBF, C=1, gamma=0.5")
		print("=" * 80)
		return BEST_KERNEL, 1.0, 0.5

	# 打乱后再切分，避免按目录顺序切分带来的偏差。
	rng = np.random.default_rng(42)
	idx = np.arange(len(features), dtype=np.int32)
	rng.shuffle(idx)
	features = features[idx]
	labels = labels[idx]

	split_idx = int(0.9 * len(features))
	split_idx = max(1, min(split_idx, len(features) - 1))
	X_train, X_test = features[:split_idx], features[split_idx:]
	Y_train, Y_test = labels[:split_idx], labels[split_idx:]

	print("=" * 80)
	print("SVM 参数组合 准确率对比表")
	print("=" * 80)
	print(f"{'序号':<6}{'C(惩罚系数)':<15}{'gamma(核系数)':<18}{'测试准确率':<15}")
	print("-" * 80)

	result_list = []
	row_id = 0
	for c in C_LIST:
		for gamma in GAMMA_LIST:
			row_id += 1
			svm = SVM(kernel=BEST_KERNEL, c=c, gamma=gamma)
			svm.train(X_train, Y_train)
			pred = svm.predict(X_test)
			acc = round(float(np.mean(pred == Y_test)), 4)
			result_list.append((c, gamma, acc))
			print(f"{row_id:<6}{c:<15}{gamma:<18}{acc:<15}")

	best_c, best_gamma, best_acc = max(result_list, key=lambda x: x[2])
	print("=" * 80)
	print(f"最优参数组合：C = {best_c}，gamma = {best_gamma}，准确率 = {best_acc:.4f}")
	print("=" * 80)
	return BEST_KERNEL, float(best_c), float(best_gamma)


def main() -> None:
	"""执行字符 SVM 的 K 折验证、最终训练、测试评估与模型保存。"""
	script_dir = Path(__file__).resolve().parent

	# 数据目录（当前工程结构）
	alnum_train = script_dir / Path("train/chars")
	alnum_test = script_dir / Path("test/chars")
	chinese_train = script_dir / Path("train/charsChinese")
	chinese_test = script_dir / Path("test/charsChinese")

	model_dir = script_dir / Path("models")
	svm_c = 1.0
	svm_gamma = 0.5
	k_folds = 5
	random_seed = 42

	alnum_class_filter = lambda name: len(name) == 1
	chinese_class_filter = lambda name: name.startswith("zh_")

	train_x, train_y = load_features_from_dir(
		alnum_train,
		label_from_class=lambda cls: ord(cls),
		class_filter=alnum_class_filter,
	)
	if train_x.size == 0:
		raise RuntimeError(f"ALNUM SVM: no training samples found in {alnum_train}")

	run_kfold_cv(
		train_x,
		train_y,
		k_folds=k_folds,
		c=svm_c,
		gamma=svm_gamma,
		title="ALNUM SVM",
		seed=random_seed,
	)

	best_kernel, best_c, best_gamma = find_best_svm_params_with_report(train_x, train_y)
	alnum_svm = SVM(kernel=best_kernel, c=best_c, gamma=best_gamma)
	alnum_svm.train(train_x, train_y)
	train_pred = alnum_svm.predict(train_x)

	print("\n[ALNUM SVM - Final Train/Test]")
	print(f"  train samples={len(train_y):<6} acc={accuracy(train_y, train_pred):.4f}")
	evaluate_optional_split(alnum_svm, alnum_test, lambda cls: ord(cls), alnum_class_filter, "test")

	model_dir.mkdir(parents=True, exist_ok=True)
	alnum_model_path = model_dir / "svm.dat"
	alnum_svm.save(str(alnum_model_path))
	print(f"  saved model -> {alnum_model_path}")

	province_to_idx = {name: i for i, name in enumerate(PROVINCES)}
	train_x, train_y = load_features_from_dir(
		chinese_train,
		label_from_class=lambda cls: province_to_idx[cls] + PROVINCE_START + 1,
		class_filter=chinese_class_filter,
	)
	if train_x.size == 0:
		raise RuntimeError(f"CHINESE SVM: no training samples found in {chinese_train}")

	run_kfold_cv(
		train_x,
		train_y,
		k_folds=k_folds,
		c=svm_c,
		gamma=svm_gamma,
		title="CHINESE SVM",
		seed=random_seed,
	)

	best_kernel_cn, best_c_cn, best_gamma_cn = find_best_svm_params_with_report(train_x, train_y)
	chinese_svm = SVM(kernel=best_kernel_cn, c=best_c_cn, gamma=best_gamma_cn)
	chinese_svm.train(train_x, train_y)
	train_pred = chinese_svm.predict(train_x)

	print("\n[CHINESE SVM - Final Train/Test]")
	print(f"  train samples={len(train_y):<6} acc={accuracy(train_y, train_pred):.4f}")
	evaluate_optional_split(
		chinese_svm,
		chinese_test,
		lambda cls: province_to_idx[cls] + PROVINCE_START + 1,
		chinese_class_filter,
		"test",
	)

	chinese_model_path = model_dir / "svmchinese.dat"
	chinese_svm.save(str(chinese_model_path))
	print(f"  saved model -> {chinese_model_path}")


if __name__ == "__main__":
	main()
