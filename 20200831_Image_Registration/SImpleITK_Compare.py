# https://www.insight-journal.org/rire/download_training_data.php
# https://itk.org/Wiki/SimpleITK/Tutorials/MICCAI2015
import SimpleITK as sitk
import matplotlib.pyplot as plt

fixed = sitk.ReadImage("images/mr_T1_01.jpg", sitk.sitkFloat32)
moving = sitk.ReadImage("images/mr_T1_01_trans.jpg", sitk.sitkFloat32)

registration_method = sitk.ImageRegistrationMethod()
# Euler2DTransform() : Euler angle로 표현된 회전을 포함한 rigid transformation
initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler2DTransform())

registration_method.SetMetricAsMeanSquares()  # 음수의 평균 제곱 사용
# step이 너무 커지지 않도록 방지하는 경사하강법
registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=4.0, minStep=.01, numberOfIterations=200)
registration_method.SetInterpolator(sitk.sitkLinear)  # 보간법 정의
registration_method.SetInitialTransform(initial_transform)
outTx1 = registration_method.Execute(fixed, moving)
# print(outTx1)
# GetOptimizerStopConditionDescription() : 실행 결과에 대한 상태 서술
# print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
# GetOptimizerIteration() : 반복횟수 반환
# print("Number of iterations: {0}".format(R.GetOptimizerIteration()))

registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)  # 서로의 상호 정보를 이용하여 이미지 registraion 진행
# step이 너무 커지지 않도록 방지하는 경사하강법
registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=4.0, minStep=.01, numberOfIterations=200)
registration_method.SetInterpolator(sitk.sitkLinear)  # 보간법 정의
registration_method.SetInitialTransform(initial_transform)
outTx2 = registration_method.Execute(fixed, moving)
# print(outTx2)
print("Optimizer stop condition: {0}".format(registration_method.GetOptimizerStopConditionDescription()))
print("Number of iterations: {0}".format(registration_method.GetOptimizerIteration()))

# sitk.WriteTransform(outTx, 'transfo_final.tfm')
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(100)

# Mean Square
resampler.SetTransform(outTx1)
out1 = resampler.Execute(moving)
moving_image_array_trans1 = sitk.GetArrayFromImage(out1)
simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out1), sitk.sitkUInt8)
cimg1_array = sitk.GetArrayFromImage(sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.))

# Mutual Information
resampler.SetTransform(outTx2)
out2 = resampler.Execute(moving)
moving_image_array_trans2 = sitk.GetArrayFromImage(out2)
simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out2), sitk.sitkUInt8)
cimg2_array = sitk.GetArrayFromImage(sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.))

fixed_image_array = sitk.GetArrayFromImage(fixed)
moving_image_array = sitk.GetArrayFromImage(moving)

plt.figure(figsize=(20, 30))
plt.gray()
plt.subplot(321), plt.imshow(fixed_image_array), plt.axis('off'), plt.title('MR-T1 Image', size=20)
plt.subplot(322), plt.imshow(moving_image_array), plt.axis('off'), plt.title('Shifted MR_T1 Image', size=20)
plt.subplot(323), plt.imshow(fixed_image_array - moving_image_array_trans1), plt.axis('off'), plt.title(
    'Difference Images (MeanSquare)', size=20)
plt.subplot(324), plt.imshow(fixed_image_array - moving_image_array_trans2), plt.axis('off'), plt.title(
    'Difference Images (MutualInformation)', size=20)
plt.subplot(325), plt.imshow(cimg1_array), plt.axis('off'), plt.title('Aligned Images (MeanSquare)', size=20)
plt.subplot(326), plt.imshow(cimg2_array), plt.axis('off'), plt.title('Aligned Images (MutualInformation)', size=20)
plt.show()
# exit()

checkerboard = sitk.CheckerBoardImageFilter()
before_reg_image = checkerboard.Execute(fixed, moving)
after_reg_image = checkerboard.Execute(fixed, out2)
plt.figure(figsize=(20,10))
plt.gray()
plt.subplot(121), plt.imshow(sitk.GetArrayFromImage(before_reg_image)), plt.axis('off'), plt.title('Checkerboard before Registration Image', size=20)
plt.subplot(122), plt.imshow(sitk.GetArrayFromImage(after_reg_image)), plt.axis('off'), plt.title('Checkerboard After Registration Image', size=20)
plt.show()