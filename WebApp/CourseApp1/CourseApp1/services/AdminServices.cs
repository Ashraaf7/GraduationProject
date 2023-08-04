using CourseApp1.data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace CourseApp1.services
{
    public interface IAdminservice
    {
        bool login(String Email, String password);
        bool changepassword(String Email, String Password);
        bool forgetPassword(String Email, String Password);
    }
    public class AdminServices : IAdminservice
    {
        public courses_dbEntities context { get; set; }
        public AdminServices()
        {
            context= new courses_dbEntities();
        }
        public bool changepassword(string Email, string Password)
        {
            throw new NotImplementedException();
        }

        public bool forgetPassword(string Email, string Password)
        {
            throw new NotImplementedException();
        }

        public bool login(string Email, string password)
        {
            return context.Admins.Where(a => a.Email == Email && a.Password==password).Any();
        }
    }
}