package com.example.catbreedsclassify;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.StrictMode;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import static com.example.catbreedsclassify.ImageUtilsExtend.getImageAbsolutePath;


public class MainActivity extends AppCompatActivity {

    //Load the tensorflow inference library
    static {
        System.loadLibrary("tensorflow_inference");
    }


    private ImageView imgiv;
    private String TAG = "tag";
    TextView resultView;
    Snackbar progressBar;
    private Button cameraBt;
    private Button photoBt;

    //需要的权限数组 读/写/相机
    private static String[] PERMISSIONS_STORAGE = {Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.CAMERA};


    //PATH TO OUR MODEL FILE AND NAMES OF THE INPUT AND OUTPUT NODES
    private final String MODEL_PATH = "file:///android_asset/incp_finetuned_stripped.pb";
    private final String INPUT_NAME = "Cast";
    private final String Placeholder_1 = "Placeholder_1";
    private final String Placeholder_2 = "Placeholder_2";
    private final String Placeholder_3 = "Placeholder_3";
    private final String OUTPUT_NAME = "Softmax_1";
    private TensorFlowInferenceInterface tf;

    //ARRAY TO HOLD THE PREDICTIONS AND FLOAT VALUES TO HOLD THE IMAGE DATA
    float[] PREDICTIONS = new float[67];
    private float[] floatValues;
    private final int[] INPUT_SIZE = {224,224,3};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        //跳转相机动态权限
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            StrictMode.VmPolicy.Builder builder = new StrictMode.VmPolicy.Builder();
            StrictMode.setVmPolicy(builder.build());
        }
        initView();


        //initialize tensorflow with the AssetManager and the Model
        tf = new TensorFlowInferenceInterface(getAssets(),MODEL_PATH);

        resultView = (TextView) findViewById(R.id.results);

        progressBar = Snackbar.make(imgiv,"PROCESSING IMAGE",Snackbar.LENGTH_INDEFINITE);


        final FloatingActionButton predict = (FloatingActionButton) findViewById(R.id.predict);
        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {


                try{

                    //READ THE IMAGE FROM ASSETS FOLDER
//                    InputStream imageStream = getAssets().open("8736168_253.jpg");
//
//                    Bitmap bitmap = BitmapFactory.decodeStream(imageStream);

                    Bitmap bitmap = ((BitmapDrawable) ((ImageView) imgiv).getDrawable()).getBitmap();

                    progressBar.show();

                    predict(bitmap);
                }
                catch (Exception e){

                }

            }
        });
    }

    //FUNCTION TO COMPUTE THE MAXIMUM PREDICTION AND ITS CONFIDENCE
    public Object[] argmax(float[] array){


        int best = -1;
        float best_confidence = 0.0f;

        for(int i = 0;i < array.length;i++){

            float value = array[i];

            if (value > best_confidence){

                best_confidence = value;
                best = i;
            }
        }

        return new Object[]{best,best_confidence};


    }


    public void predict(final Bitmap bitmap){


        //Runs inference in background thread
        new AsyncTask<Integer,Integer,Integer>(){

            @Override

            protected Integer doInBackground(Integer ...params){

                //Resize the image into 224 x 224
                Bitmap resized_image = ImageUtils.processBitmap(bitmap,224);

                //Normalize the pixels
                floatValues = ImageUtils.normalizeBitmap(resized_image,224,127.5f,1.0f);

                //Pass input into the tensorflow
                tf.feed(INPUT_NAME,floatValues,224,224,3);
                tf.feed(Placeholder_1, new boolean[]{false});
                tf.feed(Placeholder_2,new float[]{0});
                tf.feed(Placeholder_3,new float[]{0});

                //compute predictions
                tf.run(new String[]{OUTPUT_NAME});

                //copy the output into the PREDICTIONS array
                tf.fetch(OUTPUT_NAME,PREDICTIONS);

                //Obtained highest prediction
                Object[] results1 = argmax(PREDICTIONS);
                int class_index1 = (Integer) results1[0];
                float confidence1 = (Float) results1[1];
                PREDICTIONS[class_index1]=-1;

                Object[] results2 = argmax(PREDICTIONS);
                int class_index2 = (Integer) results2[0];
                float confidence2 = (Float) results2[1];
                PREDICTIONS[class_index2]=-1;

                Object[] results3 = argmax(PREDICTIONS);
                int class_index3 = (Integer) results3[0];
                float confidence3 = (Float) results3[1];



                try{

                    final String conf1 = String.valueOf(confidence1 * 100).substring(0,5);
                    //Convert predicted class index into actual label name
                   final String label1 = ImageUtils.getLabel(getAssets().open("labels.json"),class_index1);
                    final String name1 = label1.split(("\\|"))[0];
                    final String des1 = label1.split(("\\|"))[1];

                    final String conf2 = String.valueOf(confidence2 * 100).substring(0,5);
                    //Convert predicted class index into actual label name
                    final String label2 = ImageUtils.getLabel(getAssets().open("labels.json"),class_index2);
                    final String name2 = label2.split(("\\|"))[0];
                    final String des2 = label2.split(("\\|"))[1];

                    final String conf3 = String.valueOf(confidence3 * 100).substring(0,5);
                    //Convert predicted class index into actual label name
                    final String label3 = ImageUtils.getLabel(getAssets().open("labels.json"),class_index3);
                    final String name3 = label3.split(("\\|"))[0];
                    final String des3 = label3.split(("\\|"))[1];


                   //Display result on UI
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {

                            progressBar.dismiss();
                            resultView.setText(name1 + " " + conf1 + "%\n"+des1+"\n\n"+name2+" "+conf2+"%\n"+des2+"\n\n"+name3+" "+conf3+"%\n"+des3);

                        }
                    });

                }

                catch (Exception e){


                }


                return 0;
            }



        }.execute(0);

    }

    private Uri ImageUri;
    public static final int TAKE_PHOTO = 101;
    public static final int TAKE_FILE = 100;

    private void initView() {
        cameraBt = (Button) findViewById(R.id.camera_bt);
        photoBt = (Button) findViewById(R.id.photo_bt);
        imgiv = (ImageView) findViewById(R.id.img_iv);


        cameraBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //检查是否已经获得相机的权限
                if (verifyPermissions(MainActivity.this, PERMISSIONS_STORAGE[2]) == 0) {
                    Log.i(TAG, "提示是否要授权");
                    ActivityCompat.requestPermissions(MainActivity.this, PERMISSIONS_STORAGE, 3);
                } else {
                    //已经有权限
                    toCamera();  //打开相机
                }
            }
        });
        photoBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toPicture();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case TAKE_PHOTO:
                if (resultCode == RESULT_OK) {
                    try {
                        //将拍摄的照片显示出来

                        String path = ImageUri.getPath();
                        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(ImageUri));
                        System.out.println("#########TAKE_PHOTO");
//                        String path = ImageUtilsExtend.getImageAbsolutePath(MainActivity.this,ImageUri);
                        int degree = ImageUtils.getBitmapDegree(path);
                        if (degree == 0) {
                            bitmap = ImageUtils.rotateBitmapByDegree(bitmap, 90);
                        }

                        imgiv.setImageBitmap(bitmap);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                break;
            case TAKE_FILE:
                if (resultCode == RESULT_OK) {
                    try {
                        //将相册的照片显示出来
                        Uri uri_photo = data.getData();
                        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri_photo));

                        String path = ImageUtilsExtend.getImageAbsolutePath(this.getApplicationContext(),uri_photo);
                        int degree = ImageUtils.getBitmapDegree(path);
                        if (degree != 0) {
                            bitmap = ImageUtils.rotateBitmapByDegree(bitmap, degree);
                        }
                        imgiv.setImageBitmap(bitmap);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                break;
            default:
                break;
        }
    }

    /**
     * 检查是否有对应权限
     *
     * @param activity   上下文
     * @param permission 要检查的权限
     * @return 结果标识
     */
    public int verifyPermissions(Activity activity, java.lang.String permission) {
        int Permission = ActivityCompat.checkSelfPermission(activity, permission);
        if (Permission == PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "已经同意权限");
            return 1;
        } else {
            Log.i(TAG, "没有同意权限");
            return 0;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (grantResults != null && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "用户授权");
            toCamera();
        } else {
            Log.i(TAG, "用户未授权");
        }
    }

    //跳转相册
    private void toPicture() {
        Intent intent = new Intent(Intent.ACTION_PICK);  //跳转到 ACTION_IMAGE_CAPTURE
        intent.setType("image/*");
        startActivityForResult(intent, TAKE_FILE);
        Log.i(TAG, "跳转相册成功");
    }

    //跳转相机
    private void toCamera() {
        //创建File对象，用于存储拍照后的图片
//        File outputImage = new File(getExternalCacheDir(), "outputImage.jpg");
        File outputImage = new File(getExternalCacheDir(), System.currentTimeMillis() + ".jpg");
        if (outputImage.exists()) {
            outputImage.delete();
        } else {
            try {
                outputImage.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        //判断SDK版本高低，ImageUri方法不同
        if (Build.VERSION.SDK_INT >= 24) {
            ImageUri = FileProvider.getUriForFile(MainActivity.this, "com.example.catbreedsclassify.fileprovider", outputImage);
        } else {
            ImageUri = Uri.fromFile(outputImage);
        }

        //启动相机程序
        Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");
        intent.putExtra(MediaStore.EXTRA_OUTPUT, ImageUri);
        startActivityForResult(intent, TAKE_PHOTO);
    }




}
