import React, { useState, useEffect } from 'react';
import { Layout, Menu, Avatar, Button, Space, message } from 'antd';
import {
  UserOutlined,
  SettingOutlined,
  BookOutlined,
  HistoryOutlined
} from '@ant-design/icons';
import UserInfo from './components/UserInfo';
import PrivacySettings from './components/PrivacySettings';
import StudyPlan from './components/StudyPlan';
import KnowledgeTraining from './components/KnowledgeTraining';
import './index.less';
import { useNavigate } from 'react-router-dom';

const { Header, Sider, Content } = Layout;

// 从 localStorage 获取用户信息，如果没有则使用默认值
const getInitialUserInfo = () => {
  const savedUserInfo = localStorage.getItem('userInfo');
  return savedUserInfo ? JSON.parse(savedUserInfo) : {
    avatar: '',
    nickname: '用户1234',
    focusTime: 0,
    lastLoginTime: '',
    phone: '',
    email: ''
  };
};  

const UserCenter = () => {
  const [selectedMenu, setSelectedMenu] = useState('userInfo');
  const [userInfo, setUserInfo] = useState(getInitialUserInfo);
  const navigate = useNavigate();

  // 当 userInfo 更新时，保存到 localStorage
  useEffect(() => {
    localStorage.setItem('userInfo', JSON.stringify(userInfo));
  }, [userInfo]);

  // 更新用户信息的处理函数
  const handleUpdateUserInfo = (newInfo) => {
    setUserInfo(newInfo);
  };

  // 渲染对应的内容组件
  const renderContent = () => {
    switch(selectedMenu) {
      case 'userInfo':
        return <UserInfo userInfo={userInfo} onUpdate={handleUpdateUserInfo} />;
      case 'privacy':
        return <PrivacySettings />;
      case 'studyPlan':
        return <StudyPlan />;
      case 'knowledge':
        return <KnowledgeTraining />;
      default:
        return <UserInfo userInfo={userInfo} onUpdate={handleUpdateUserInfo} />;
    }
  };

  const handleLogout = () => {
    // 清除所有相关的状态
    localStorage.clear();  // 清除所有本地存储
    // 或者只清除特定的项
    // localStorage.removeItem('isLoggedIn');
    // localStorage.removeItem('userInfo');
    
    message.success('退出登录成功');
    navigate('/login', { replace: true });
  };

  return (
    <Layout className="user-center">
      <Header className="header">
        <div className="logo">学习助手</div>
        <Menu mode="horizontal" theme="dark">
          <Menu.Item key="home" onClick={() => navigate('/')}>首页</Menu.Item>
          <Menu.Item key="plan" onClick={() => navigate('/study-plan')}>学习计划</Menu.Item>
          <Menu.Item key="feedback" onClick={() => navigate('/feedback')}>反馈</Menu.Item>
          <Menu.Item key="user" onClick={() => navigate('/user')}>个人中心</Menu.Item>
        </Menu>
        <div className="header-right">
          <Space>
            <Button type="primary" ghost onClick={() => navigate('/register')}>
              注册
            </Button>
            <Button type="primary" onClick={handleLogout}>
              退出登录
            </Button>
          </Space>
        </div>
      </Header>
      
      <Layout>
        <Sider width={200} className="site-sider">
          <div className="user-brief">
            <Avatar size={64} icon={<UserOutlined />} src={userInfo.avatar} />
            <div className="user-info">
              <div className="nickname">{userInfo.nickname}</div>
              <div className="focus-time">专注时长：{userInfo.focusTime}分钟</div>
            </div>
          </div>
          <Menu
            mode="inline"
            selectedKeys={[selectedMenu]}
            onSelect={({key}) => setSelectedMenu(key)}
            className="side-menu"
          >
            <Menu.Item key="userInfo" icon={<UserOutlined />}>
              个人信息
            </Menu.Item>
            <Menu.Item key="privacy" icon={<SettingOutlined />}>
              隐私设置
            </Menu.Item>
            <Menu.Item key="studyPlan" icon={<BookOutlined />}>
              学习计划
            </Menu.Item>
            <Menu.Item key="knowledge" icon={<HistoryOutlined />}>
              知识点掌握情况
            </Menu.Item>
          </Menu>
        </Sider>
        
        <Content className="main-content">
          {renderContent()}
        </Content>
      </Layout>
    </Layout>
  );
};

export default UserCenter; 