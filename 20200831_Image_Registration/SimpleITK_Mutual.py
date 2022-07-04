# https://stackoverflow.com/questions/41692063/what-is-the-difference-between-image-registration-and-image-alignment
# https://www.insight-journal.org/rire/download_training_data.php
# https://itk.org/Wiki/SimpleITK/Tutorials/MICCAI2015
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# <class 'SimpleITK.SimpleITK.Image'>
fixed_image = sitk.ReadImage("images/ct_scan_11.jpg", sitk.sitkFloat32)
moving_image = sitk.ReadImage("images/mr_T1_01.jpg", sitk.sitkFloat32)

fixed_image_array = sitk.GetArrayFromImage(fixed_image)  # Image형식에서 numpy형식으로
moving_image_array = sitk.GetArrayFromImage(moving_image)  # Image형식에서 numpy형식으로

np.random.seed(2)
# <class 'SimpleITK.SimpleITK.ImageRegistrationMethod'>
registration_method = sitk.ImageRegistrationMethod()  # registraion 객체

# <class 'SimpleITK.SimpleITK.Transform'>
# CenteredTransformInitializer() : 중앙을 중심으로 하는 Transform 객체
# Similarity2DTransform() : Euler angle로 표현된 회전을 포함한 rigid 변환에 scaling 추가
initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Similarity2DTransform())

# Similarity metric settings.
# 서로의 상호 정보를 이용하여 이미지 registraion 진행
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)  # sampling하기 위한 방법선언 -> RANDOM
registration_method.SetMetricSamplingPercentage(0.5)  # metric 평가를 위한 pixel의 퍼센트 정의, 기본값 1.0
registration_method.SetInterpolator(sitk.sitkLinear)  # 보간법 정의

# Optimizer settings.
# 경사하강법
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                  convergenceMinimumValue=1e-6, convergenceWindowSize=10)
# 단계별 voxel 이동의 최대값을 이용해 transform parameters의 크기 조절
registration_method.SetOptimizerScalesFromPhysicalShift()

# 다양한 해상도를 위한 세팅
registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# initial transform을 정해주고, 제자리 최적화가 이루어지지 않게 한다(여러번 돌림)
registration_method.SetInitialTransform(initial_transform, inPlace=False)

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
