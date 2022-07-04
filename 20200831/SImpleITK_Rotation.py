import SimpleITK as sitk
import matplotlib.pyplot as plt

# <class 'SimpleITK.SimpleITK.Image'>
fixed_image = sitk.ReadImage("images/ct_scan_11.jpg", sitk.sitkFloat32)
moving_image = sitk.ReadImage("images/mr_T1_01_rot.png", sitk.sitkFloat32)

fixed_image_array = sitk.GetArrayFromImage(fixed_image)  # Image형식에서 numpy형식으로
moving_image_array = sitk.GetArrayFromImage(moving_image)  # Image형식에서 numpy형식으로

# <class 'SimpleITK.SimpleITK.ImageRegistrationMethod'>
registration_method = sitk.ImageRegistrationMethod()  # registraion 객체

# <class 'SimpleITK.SimpleITK.Transform'>
initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler2DTransform())

# Similarity metric settings.
# 서로의 상호 정보를 이용하여 이미지 registraion 진행
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetInterpolator(sitk.sitkLinear)  # 보간법 정의

# Optimizer settings.
registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=4.0, minStep=.01, numberOfIterations=200)

# 다양한 해상도를 위한 세팅
registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# initial transform을 정해주고, 제자리 최적화가 이루어지지 않게 한다(여러번 돌림)
registration_method.SetInitialTransform(initial_transform)

# <class 'SimpleITK.SimpleITK.Transform'>
final_transform = registration_method.Execute(fixed_image, moving_image)

# <class 'SimpleITK.SimpleITK.ResampleImageFilter'>
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed_image)  # 참조할 이미지
resampler.SetInterpolator(sitk.sitkLinear)  # 보간법정의
resampler.SetDefaultPixelValue(100)  # pixel이 이미지 밖으로 나갔을때 픽셀값 조정
resampler.SetTransform(final_transform)  # 얻어진 transform 설정

# out : <class 'SimpleITK.SimpleITK.Image'>
out = resampler.Execute(moving_image)  # 움직일 이미지를 input으로 하여 옮겨진 이미지를 out에 할당
simg1 = sitk.Cast(sitk.RescaleIntensity(fixed_image), sitk.sitkUInt8)  # uint8로 정규화
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)  # 이미지 합치기

plt.figure(figsize=(20, 10))
plt.gray()
plt.subplot(231), plt.imshow(fixed_image_array), plt.axis('off'), plt.title('CT-Scan Image')
plt.subplot(232), plt.imshow(moving_image_array), plt.axis('off'), plt.title('MRI-T1 Image')
plt.subplot(233), plt.imshow(fixed_image_array//2. + moving_image_array//2.), plt.axis('off'), plt.title('Initial Alignment')
plt.subplot(234), plt.imshow(sitk.GetArrayFromImage(out)), plt.axis('off'), plt.title('Transformed MRI-T1 Image')
plt.subplot(235), plt.imshow(sitk.GetArrayFromImage(simg1//2.+simg2//2.)), plt.axis('off'), plt.title('Transformed Alignment')
plt.subplot(236), plt.imshow(sitk.GetArrayFromImage(cimg)), plt.axis('off'), plt.title('Composed')
plt.show()
