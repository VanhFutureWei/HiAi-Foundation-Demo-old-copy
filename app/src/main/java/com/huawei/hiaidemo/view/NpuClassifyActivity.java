/*
 *@file NpuClassifyActivity.java
 *
 * Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

package com.huawei.hiaidemo.view;

import android.Manifest;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.Toast;

import com.huawei.hiaidemo.R;
import com.huawei.hiaidemo.adapter.ClassifyAdapter;
import com.huawei.hiaidemo.bean.ClassifyItemModel;
import com.huawei.hiaidemo.bean.ModelInfo;
import com.huawei.hiaidemo.utils.ModelManager;
import com.huawei.hiaidemo.utils.TestUtils;
import com.huawei.hiaidemo.utils.Untils;

import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;

import static com.huawei.hiaidemo.utils.Constant.GALLERY_REQUEST_CODE;
import static com.huawei.hiaidemo.utils.Constant.IMAGE_CAPTURE_REQUEST_CODE;


public abstract class NpuClassifyActivity extends AppCompatActivity{
    private static final String TAG = NpuClassifyActivity.class.getSimpleName();
    protected List<ClassifyItemModel> items;

    protected RecyclerView rv;

    protected AssetManager mgr;

    protected String[] predictedClass =  new String[3];

    protected Bitmap initClassifiedImg;

    protected ClassifyAdapter adapter;

    protected Button btnGallery;
    protected Button btnCamera;

    protected ModelInfo selectedModel;

    protected ArrayList<ModelInfo> modelList;

    protected Vector<String> word_label =  new Vector<String>();

    protected float inferenceTime;

    protected float[] outputData;

    protected ArrayList<float[]> outputDataList;

    private int[] imageBitmapPixels = new int[244*244];
    private float[] imageNormalizedPixels= new float[244*244*3];

    public static float grandTime=0.f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getSupportActionBar().hide();
        setContentView(R.layout.activity_npu_classify);

        items = new ArrayList<>();

        mgr = getResources().getAssets();

        initView();

        modelList = (ArrayList<ModelInfo>)getIntent().getSerializableExtra("demoModelList");

        modelList = loadModel(modelList);

        ArrayList<String> modelNames = new ArrayList<String>();
        for(ModelInfo modelInfo : modelList) {
            modelNames.add(modelInfo.getOfflineModelName());
        }

        //use default model
        selectedModel = modelList.get(0);
        //overwritten by specified model
        for(ModelInfo model : modelList) {
            if(model.getOfflineModelName().equals(MainActivity.selectedModelName)) {
                selectedModel = model;
                break;
            }
        }

        preProcess();

    }

    private void setHeaderView(RecyclerView view) {
        View header = LayoutInflater.from(this).inflate(R.layout.recyclerview_hewader, view, false);

        btnGallery = header.findViewById(R.id.btn_gallery);
        btnCamera = header.findViewById(R.id.btn_camera);
        adapter.setHeaderView(header);
    }

    private void initView() {
        rv = (RecyclerView) findViewById(R.id.rv);
        LinearLayoutManager manager = new LinearLayoutManager(this);
        rv.setLayoutManager(manager);

        adapter = new ClassifyAdapter(items);
        rv.setAdapter(adapter);

        setHeaderView(rv);

        /*
        btnGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                checkStoragePermission();
            }
        });*/
        btnGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                batchImageRun();
            }
        });

        btnCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                checkStoragePermission();
            }
        });
    }

    private void checkStoragePermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA},
                    GALLERY_REQUEST_CODE);
        } else {
            //selectedModel may be changed,so reassign.
            for(ModelInfo model : modelList) {
                if(model.getOfflineModelName().equals(MainActivity.selectedModelName)) {
                    selectedModel = model;
                    Toast.makeText(NpuClassifyActivity.this, "Run Model:"+MainActivity.selectedModelName, Toast.LENGTH_SHORT).show();
                    break;
                }
            }
            chooseImageAndClassify();
        }
    }


    private void batchImageRun(){
        String[] valBatchImages = new String[0];
        try {
            valBatchImages = getAssets().list("val_batch");
        } catch (IOException e) {
            e.printStackTrace();
        }
        int count=0;
        for (String valImagePath : valBatchImages) {
            //Log.d("DUMPLOG", valImagePath);
            Bitmap bitmap = TestUtils.getBitmapFromAsset(getAssets(), "val_batch/" + valImagePath);
            count++;

            Log.d(TAG, String.valueOf(bitmap.getWidth())+" "+String.valueOf(bitmap.getHeight())+" "+String.valueOf(bitmap.getByteCount())+" ");

            Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            initClassifiedImg = Bitmap.createScaledBitmap(rgba, selectedModel.getInput_W(), selectedModel.getInput_H(), true);

            byte[] inputData;
            ByteBuffer byteBufferToClassify;

            if(selectedModel.getUseAIPP()){
                inputData = Untils.getPixelsAIPP(selectedModel.getFramework(),initClassifiedImg,selectedModel.getInput_W(),selectedModel.getInput_H());
            }else {
                Bitmap toClassify = ThumbnailUtils.extractThumbnail(initClassifiedImg, selectedModel.getInput_W(),selectedModel.getInput_H());
                byteBufferToClassify = bitmapToModelsMatchingByteBuffer(toClassify);
                byteBufferToClassify.flip();
                inputData = getByteArrayFromByteBuffer(byteBufferToClassify);
                Log.d(TAG, "batchImageRun: " + inputData.length);
            }
            ArrayList<byte[]> inputDataList = new ArrayList<>();
            inputDataList.add(inputData);

            runModel(selectedModel,inputDataList);
        }

        Log.d("TESTING","Average time: "+(grandTime/count));
        Toast.makeText(this, "Average time: "+(grandTime/count), Toast.LENGTH_SHORT).show();

    }


    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA},
                    IMAGE_CAPTURE_REQUEST_CODE);
        } else {
            //selectedModel may be changed,so reassign.
            for(ModelInfo model : modelList) {
                if(model.getOfflineModelName().equals(MainActivity.selectedModelName)) {
                    selectedModel = model;
                    Toast.makeText(NpuClassifyActivity.this, "Run Model:"+MainActivity.selectedModelName, Toast.LENGTH_SHORT).show();
                    break;
                }
            }
            takePictureAndClassify();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == GALLERY_REQUEST_CODE &&
                grantResults[1] == PackageManager.PERMISSION_GRANTED) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                chooseImageAndClassify();
            } else {
                Toast.makeText(NpuClassifyActivity.this, "Permission Denied", Toast.LENGTH_SHORT).show();
            }
        }

        if (requestCode == IMAGE_CAPTURE_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED &&
                    grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                takePictureAndClassify();
            } else {
                Toast.makeText(NpuClassifyActivity.this, "Permission Denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void takePictureAndClassify() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, IMAGE_CAPTURE_REQUEST_CODE);
        }
    }

    private void chooseImageAndClassify() {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, GALLERY_REQUEST_CODE);
    }

    /*
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            switch (requestCode) {
                case GALLERY_REQUEST_CODE:
                    Bitmap bitmap;
                    //ContentResolver resolver = getContentResolver();
                    Uri originalUri = data.getData();

                    //bitmap = MediaStore.Images.Media.getBitmap(resolver, originalUri);
                    String[] proj = {MediaStore.Images.Media.DATA};
                    Cursor cursor = managedQuery(originalUri, proj, null, null, null);
                    cursor.moveToFirst();

                    int columnIndex = cursor.getColumnIndex(proj[0]);
                    String picturePath = cursor.getString(columnIndex);
                    //cursor.close();

                    //Log.d("TESTING", picturePath.toString());
                    bitmap=BitmapFactory.decodeFile(picturePath);

                    Log.d(TAG, String.valueOf(bitmap.getWidth())+" "+String.valueOf(bitmap.getHeight())+" "+String.valueOf(bitmap.getByteCount())+" ");

                    Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    initClassifiedImg = Bitmap.createScaledBitmap(rgba, selectedModel.getInput_W(), selectedModel.getInput_H(), true);

                    byte[] inputData;
                    ByteBuffer byteBufferToClassify;

                    if(selectedModel.getUseAIPP()){
                        inputData = Untils.getPixelsAIPP(selectedModel.getFramework(),initClassifiedImg,selectedModel.getInput_W(),selectedModel.getInput_H());
                    }else {
                        Bitmap toClassify = ThumbnailUtils.extractThumbnail(initClassifiedImg, selectedModel.getInput_W(),selectedModel.getInput_H());
                        byteBufferToClassify = bitmapToModelsMatchingByteBuffer(toClassify);
                        byteBufferToClassify.flip();
                        inputData = getByteArrayFromByteBuffer(byteBufferToClassify);
                    }
                    ArrayList<byte[]> inputDataList = new ArrayList<>();
                    inputDataList.add(inputData);

                    runModel(selectedModel,inputDataList);


                    Log.d("TESTING","Average time: "+(grandTime));
                    grandTime=0;
                    break;
                case IMAGE_CAPTURE_REQUEST_CODE:
                    Bundle extras = data.getExtras();
                    Bitmap imageBitmap = (Bitmap) extras.get("data");
                    //rgba = imageBitmap.copy(Bitmap.Config.ARGB_8888, true);
                    //initClassifiedImg = Bitmap.createScaledBitmap(rgba, selectedModel.getInput_W(), selectedModel.getInput_H(), true);
                    //byte[] inputData;
                    if(selectedModel.getUseAIPP()){
                        inputData = Untils.getPixelsAIPP(selectedModel.getFramework(),initClassifiedImg,selectedModel.getInput_W(),selectedModel.getInput_H());
                    }else {
                        inputData = Untils.getPixels(selectedModel.getFramework(),initClassifiedImg,selectedModel.getInput_W(),selectedModel.getInput_H());

                    }
                    inputDataList = new ArrayList<>();
                    inputDataList.add(inputData);
                    Log.i(TAG,"inputData.length is :"+inputData.length+"");
                    runModel(selectedModel,inputDataList);
                 break;

                default:
                    break;
            }
        } else {
            Toasxt.makeText(NpuClassifyActivity.this,
                    "Return without selecting pictures|Gallery has no pictures|Return without taking pictures", Toast.LENGTH_SHORT).show();
        }

    }*/


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) switch (requestCode) {
            case GALLERY_REQUEST_CODE:

                Bitmap bitmap;
                //ContentResolver resolver = getContentResolver();
                Uri originalUri = data.getData();

                //bitmap = MediaStore.Images.Media.getBitmap(resolver, originalUri);
                String[] proj = {MediaStore.Images.Media.DATA};
                Cursor cursor = managedQuery(originalUri, proj, null, null, null);
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndex(proj[0]);
                String picturePath = cursor.getString(columnIndex);
                //cursor.close();
                //Log.d("TESTING", picturePath.toString());
                bitmap= BitmapFactory.decodeFile(picturePath);
                Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);


                initClassifiedImg = Bitmap.createScaledBitmap(rgba, selectedModel.getInput_W(), selectedModel.getInput_H(), true);
                byte[] inputData = {};
                if(selectedModel.getUseAIPP()){
                    inputData = Untils.getPixelsAIPP(selectedModel.getFramework(),initClassifiedImg,selectedModel.getInput_W(),selectedModel.getInput_H());
                }else {
                    inputData = Untils.getPixels(selectedModel.getFramework(),initClassifiedImg,selectedModel.getInput_W(),selectedModel.getInput_H());
                }
                ArrayList<byte[]> inputDataList = new ArrayList<>();
                inputDataList.add(inputData);
                Log.d(TAG,"inputData.length is :"+inputData.length+"");
                runModel(selectedModel,inputDataList);

                break;
            case IMAGE_CAPTURE_REQUEST_CODE:

                Bundle extras = data.getExtras();
                Bitmap imageBitmap = (Bitmap) extras.get("data");
                Bitmap rgba2 = imageBitmap.copy(Bitmap.Config.ARGB_8888, true);
                initClassifiedImg = Bitmap.createScaledBitmap(rgba2, selectedModel.getInput_W(), selectedModel.getInput_H(), true);
                byte[] inputData2 = {};
                if(selectedModel.getUseAIPP()){
                    inputData2 = Untils.getPixelsAIPP(selectedModel.getFramework(),initClassifiedImg,selectedModel.getInput_W(),selectedModel.getInput_H());
                }else {
                    inputData2 = Untils.getPixels(selectedModel.getFramework(),initClassifiedImg,selectedModel.getInput_W(),selectedModel.getInput_H());
                }
                ArrayList<byte[]> inputDataList2 = new ArrayList<>();
                inputDataList2.add(inputData2);
                Log.i(TAG,"inputData.length is :"+inputData2.length+"");
                runModel(selectedModel,inputDataList2);
                break;

            default:
                break;
        }
        else {
            Toast.makeText(NpuClassifyActivity.this,
                    "Return without selecting pictures|Gallery has no pictures|Return without taking pictures", Toast.LENGTH_SHORT).show();
        }

    }

    private static byte[] getByteArrayFromByteBuffer(ByteBuffer byteBuffer) {
        byte[] bytesArray = new byte[byteBuffer.remaining()];
        byteBuffer.get(bytesArray, 0, bytesArray.length);
        return bytesArray;
    }

    private ByteBuffer bitmapToModelsMatchingByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(selectedModel.getInput_W()*selectedModel.getInput_H()*3*4);

        Log.d(TAG, "bitmapToModelsMatchingByteBuffer: " + selectedModel.getInput_W()*selectedModel.getInput_H()*3*4);

        float rVals[] = new float[selectedModel.getInput_W()*selectedModel.getInput_H()];
        float gVals[] = new float[selectedModel.getInput_W()*selectedModel.getInput_H()];
        float bVals[] = new float[selectedModel.getInput_W()*selectedModel.getInput_H()];
        int index =0;

        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[selectedModel.getInput_W() * selectedModel.getInput_H()];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < selectedModel.getInput_W(); ++i) {
            for (int j = 0; j < selectedModel.getInput_H(); ++j) {
                int pixelVal = intValues[pixel++];
                float[] channelVal = pixelToChannelValues(pixelVal);
                rVals[index]=channelVal[0];
                bVals[index]=channelVal[1];
                gVals[index]=channelVal[2];
                index++;
                /*
                for (float channelVal : pixelToChannelValues(pixelVal)) {
                    Log.d("TESTING", String.valueOf(channelVal));
                    byteBuffer.putFloat(channelVal);
                }*/
            }
        }

        for (float r:rVals)
            byteBuffer.putFloat(r);
        for (float g:gVals)
            byteBuffer.putFloat(g);
        for (float b:bVals)
            byteBuffer.putFloat(b);

        return byteBuffer;
    }

    private float[] pixelToChannelValues(int pixel) {
        float[] rgbVals = new float[3];
        float imageMean = 0.f;
        float imageStd = 255.f;
        rgbVals[0] = ((((pixel >> 16) & 0xFF) - imageMean) / imageStd);
        rgbVals[1] = ((((pixel >> 8) & 0xFF) - imageMean) / imageStd);
        rgbVals[2] = ((((pixel) & 0xFF) - imageMean) / imageStd);
        return rgbVals;
    }

    protected void preProcess() {
        byte[] labels;
        try {
            Log.i(TAG, "modelList size: " + modelList.size());
            InputStream assetsInputStream = getAssets().open(modelList.get(0).getOnlineModelLabel());
            int available = assetsInputStream.available();
            labels = new byte[available];
            assetsInputStream.read(labels);
            assetsInputStream.close();
            String words = new String(labels);
            String[] contens = words.split("\n");

            for(String conten:contens){
                word_label.add(conten);
            }
            Log.i(TAG, "initLabel size: " + word_label.size());
        }catch (Exception e){
            Log.e(TAG,e.getMessage());
        }

    }
    protected void postProcess(float[] outputData){
        if(outputData != null){
            int[] max_index = new int[3];
            double[] max_num = new double[3];
            int maxLength = outputData.length;

            //for(int i=0; i<outputData.length;i++){
            //    Log.i("DUMPLOG",i+": "+ String.valueOf(outputData[i]));
            //}


            if(maxLength > word_label.size()){
                maxLength = word_label.size();
            }
            for (int i = 0; i < maxLength; i++) {
                double tmp = outputData[i];
                int tmp_index = i;
                for (int j = 0; j < 3; j++) {
                    if (tmp > max_num[j]) {
                        max_index[j] =tmp_index;
                        tmp += max_num[j];
                        max_num[j] = tmp - max_num[j];
                        tmp -= max_num[j];
                    }
                }
            }

            Log.i("DUMPLOG", word_label.get(max_index[0]));

            predictedClass[0] = word_label.get(max_index[0]) + " - " + max_num[0] * 100 +"%\n";
            predictedClass[1] = word_label.get(max_index[1]) + " - " + max_num[1] * 100 +"%\n"+
                    word_label.get(max_index[2]) + " - " + max_num[2] * 100 +"%\n";
            predictedClass[2] ="inference time:" +inferenceTime+ "ms\n";
            for(String res : predictedClass) {
                Log.i(TAG, res);
            }

            items.add(new ClassifyItemModel(predictedClass[0], predictedClass[1], predictedClass[2], initClassifiedImg));
            adapter.notifyDataSetChanged();
            //for(int i=0; i<outputData.length;i++)
            //  Log.i("DUMPLOG", "Classification/ Percent: "+ i+" - " + outputData[i]  );


        }else {
            Toast.makeText(NpuClassifyActivity.this,
                    "run model fail.", Toast.LENGTH_SHORT).show();
        }
    }

    protected abstract void runModel(ModelInfo modelInfo, ArrayList<byte[]> inputDataList);

    protected abstract ArrayList<ModelInfo> loadModel(ArrayList<ModelInfo> modelInfo);

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }
}
